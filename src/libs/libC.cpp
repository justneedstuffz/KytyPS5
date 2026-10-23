#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/hostMemory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/guestPrintf.h"
#include "libs/libs.h"
#include "libs/vaContext.h"
#include "loader/runtimeLinker.h"
#include "loader/symbolDatabase.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fmt/format.h>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs {

namespace LibKernel {
void KernelDispatchPendingSignalForCurrentThread();
} // namespace LibKernel

namespace LibC {

LIB_VERSION("libc", 1, "libc", 1, 1);

static uint32_t g_need_flag = 1;

using cxa_destructor_func_t = KYTY_SYSV_ABI void (*)(void*);
using atexit_func_t         = KYTY_SYSV_ABI void (*)();

struct CxaDestructor {
	cxa_destructor_func_t destructor_func;
	void*                 destructor_object;
	void*                 module_id;
};

struct CContext {
	std::list<CxaDestructor> cxa;
	std::list<atexit_func_t> atexit;
};

using InitEnvParams = Loader::EntryParams;

static int                g_argc = 0;
static const char* const* g_argv = nullptr;
static const char* const* g_envp = nullptr;

int GetArgc() {
	return g_argc;
}

const char** GetArgv() {
	return const_cast<const char**>(g_argv);
}

static KYTY_SYSV_ABI void exit(int code) {
	PRINT_NAME();

	::exit(code);
}

static void PrintAbortStringCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	const auto* ptr = reinterpret_cast<const char*>(static_cast<uintptr_t>(addr));
	char        buf[257] {};
	size_t      len = 0;

	for (; len < sizeof(buf) - 1 && Graphics::HostMemoryIsReadable(addr + len); len++) {
		const auto c = static_cast<unsigned char>(ptr[len]);
		if (c == 0) {
			break;
		}
		if (!std::isprint(c) && c != '\n' && c != '\r' && c != '\t') {
			return;
		}
		buf[len] = static_cast<char>(c);
	}

	if (len >= 4) {
		LOGF("\t %s string = \"%s\"\n", name, buf);
	}
}

static void PrintAbortWideStringCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	const auto* ptr = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(addr));
	char        buf[257] {};
	size_t      len = 0;

	for (;
	     len < sizeof(buf) - 1 && Graphics::HostMemoryIsReadable(addr + len * sizeof(uint16_t) + 1);
	     len++) {
		const auto c = ptr[len];
		if (c == 0) {
			break;
		}
		if (c > 0x7f ||
		    (!std::isprint(static_cast<unsigned char>(c)) && c != '\n' && c != '\r' && c != '\t')) {
			return;
		}
		buf[len] = static_cast<char>(c);
	}

	if (len >= 4) {
		LOGF("\t %s u16 = \"%s\"\n", name, buf);
	}
}

static void PrintAbortBytesCandidate(const char* name, uint64_t addr) {
	if (!Graphics::HostMemoryIsReadable(addr)) {
		return;
	}

	uint8_t bytes[64] {};
	size_t  len = 0;

	for (; len < sizeof(bytes) && Graphics::HostMemoryIsReadable(addr + len); len++) {
		bytes[len] = *reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(addr + len));
	}

	if (len == 0) {
		return;
	}

	LOGF("\t %s bytes =", name);
	for (size_t i = 0; i < len; i++) {
		LOGF(" %02" PRIx8, bytes[i]);
	}
	LOGF("\n");
}

static void PrintAbortPointerCandidate(const char* name, uint64_t addr) {
	PrintAbortStringCandidate(name, addr);
	PrintAbortWideStringCandidate(name, addr);
	PrintAbortBytesCandidate(name, addr);
}

static bool LooksLikeAbortPointer(uint64_t addr) {
	return Graphics::HostMemoryIsReadable(addr) && addr >= 0x10000;
}

static void PrintAbortPointerArrayCandidate(const char* name, uint64_t addr) {
	if (!LooksLikeAbortPointer(addr)) {
		return;
	}

	for (int i = 0; i < 8; i++) {
		const auto slot_addr = addr + static_cast<uint64_t>(i) * sizeof(uint64_t);
		if (!Graphics::HostMemoryIsReadable(slot_addr)) {
			break;
		}

		const auto value = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(slot_addr));
		if (!LooksLikeAbortPointer(value)) {
			continue;
		}

		LOGF("\t %s[%d] = 0x%016" PRIx64 "\n", name, i, value);
		const auto child_name = fmt::format("{}[{}]", name, i);
		PrintAbortPointerCandidate(child_name.c_str(), value);
	}
}

[[noreturn]] static KYTY_SYSV_ABI void abort(uint64_t arg0, uint64_t arg1, uint64_t arg2,
                                             uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	uint64_t rbp = 0;
	uint64_t rsp = 0;
	// This assembly is x86-64-specific, not Windows-specific.
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	asm volatile("movq %%rsp, %0" : "=r"(rsp));
#endif

	const auto ret = reinterpret_cast<uint64_t>(__builtin_return_address(0));

	LOGF("Guest abort diagnostics:\n"
	     "\t return = 0x%016" PRIx64 "\n"
	     "\t rbp    = 0x%016" PRIx64 "\n"
	     "\t rsp    = 0x%016" PRIx64 "\n"
	     "\t arg0   = 0x%016" PRIx64 "\n"
	     "\t arg1   = 0x%016" PRIx64 "\n"
	     "\t arg2   = 0x%016" PRIx64 "\n"
	     "\t arg3   = 0x%016" PRIx64 "\n"
	     "\t arg4   = 0x%016" PRIx64 "\n"
	     "\t arg5   = 0x%016" PRIx64 "\n",
	     ret, rbp, rsp, arg0, arg1, arg2, arg3, arg4, arg5);

	PrintAbortPointerCandidate("arg0", arg0);
	PrintAbortPointerCandidate("arg1", arg1);
	PrintAbortPointerCandidate("arg2", arg2);
	PrintAbortPointerCandidate("arg3", arg3);
	PrintAbortPointerCandidate("arg4", arg4);
	PrintAbortPointerCandidate("arg5", arg5);
	PrintAbortPointerArrayCandidate("arg0", arg0);
	PrintAbortPointerArrayCandidate("arg1", arg1);
	PrintAbortPointerArrayCandidate("arg2", arg2);
	PrintAbortPointerArrayCandidate("arg3", arg3);
	PrintAbortPointerArrayCandidate("arg4", arg4);
	PrintAbortPointerArrayCandidate("arg5", arg5);

	if (rbp != 0) {
		Common::Singleton<Loader::RuntimeLinker>::Instance()->StackTrace(rbp, rsp);
	}

	if (rsp != 0) {
		LOGF("Guest abort stack words:\n");
		auto* linker = Common::Singleton<Loader::RuntimeLinker>::Instance();
		for (int i = 0; i < 16; i++) {
			const auto addr = rsp + static_cast<uint64_t>(i) * sizeof(uint64_t);
			if (!Graphics::HostMemoryIsReadable(addr)) {
				break;
			}
			const auto value   = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(addr));
			auto*      program = linker->FindProgramByAddr(value);
			if (program != nullptr) {
				auto module_name = Common::PathToString(program->file_name.filename());
				LOGF("\t [%02d] 0x%016" PRIx64 " %s+0x%016" PRIx64 "\n", i, value,
				     module_name.c_str(), value - program->base_vaddr);
			} else {
				LOGF("\t [%02d] 0x%016" PRIx64 "\n", i, value);
			}
			PrintAbortPointerCandidate(fmt::format("stack[{:02d}]", i).c_str(), value);
			PrintAbortPointerArrayCandidate(fmt::format("stack[{:02d}]", i).c_str(), value);
		}
	}

	EXIT("Guest abort()\n");
	std::abort();
}

static KYTY_SYSV_ABI int* libc_error() {
	PRINT_NAME();

	return Posix::GetErrorAddr();
}

static KYTY_SYSV_ABI void init_env(const InitEnvParams* params) {
	PRINT_NAME();

	if (params == nullptr) {
		g_argc = 0;
		g_argv = nullptr;
		g_envp = nullptr;
		return;
	}

	EXIT_NOT_IMPLEMENTED(params->argc < 0);

	g_argc = params->argc;
	g_argv = params->argv;
	g_envp = params->argv + params->argc + 1;

	LOGF("\t argc = %d\n"
	     "\t argv = 0x%016" PRIx64 "\n"
	     "\t envp = 0x%016" PRIx64 "\n",
	     g_argc, reinterpret_cast<uint64_t>(g_argv), reinterpret_cast<uint64_t>(g_envp));

	for (int i = 0; i < g_argc; i++) {
		LOGF("\t argv[%d] = %s\n", i, g_argv[i] != nullptr ? g_argv[i] : "<null>");
	}
}

static KYTY_SYSV_ABI int atexit(atexit_func_t func) {
	PRINT_NAME();

	if (func != nullptr) {
		Common::Singleton<CContext>::Instance()->atexit.push_front(func);
	}

	return 0;
}

static KYTY_SYSV_ABI int libc_printf(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

	PRINT_NAME();

	return GetGuestPrintfCtxFunc()(&ctx);
}

static KYTY_SYSV_ABI int puts(const char* s) {
	PRINT_NAME();

	return GetGuestPrintfStdFunc()("%s\n", s);
}

static KYTY_SYSV_ABI int setenv(const char* name, const char* value, int overwrite) {
	PRINT_NAME();

	LOGF("\t name      = %s\n"
	     "\t value     = %s\n"
	     "\t overwrite = %d\n",
	     (name != nullptr ? name : "<null>"), (value != nullptr ? value : "<null>"), overwrite);

	if (name == nullptr || value == nullptr || name[0] == '\0' ||
	    std::strchr(name, '=') != nullptr) {
		*Posix::GetErrorAddr() = Posix::POSIX_EINVAL;
		return -1;
	}

	// Guest environment changes must not leak into emulator host libraries.
	return 0;
}

static KYTY_SYSV_ABI int64_t libc_time(int64_t* timer) {
	const auto now = static_cast<int64_t>(std::time(nullptr));
	if (timer != nullptr) {
		*timer = now;
	}
	return now;
}

static KYTY_SYSV_ABI double libc_difftime(int64_t time1, int64_t time0) {
	return std::difftime(static_cast<std::time_t>(time1), static_cast<std::time_t>(time0));
}

struct GuestTm {
	int tm_sec;
	int tm_min;
	int tm_hour;
	int tm_mday;
	int tm_mon;
	int tm_year;
	int tm_wday;
	int tm_yday;
	int tm_isdst;
};

static_assert(sizeof(GuestTm) == 36);

static GuestTm ToGuestTm(const std::tm& time) {
	return {time.tm_sec,  time.tm_min,  time.tm_hour, time.tm_mday, time.tm_mon,
	        time.tm_year, time.tm_wday, time.tm_yday, time.tm_isdst};
}

static std::tm ToHostTm(const GuestTm& time) {
	std::tm result {};
	result.tm_sec   = time.tm_sec;
	result.tm_min   = time.tm_min;
	result.tm_hour  = time.tm_hour;
	result.tm_mday  = time.tm_mday;
	result.tm_mon   = time.tm_mon;
	result.tm_year  = time.tm_year;
	result.tm_wday  = time.tm_wday;
	result.tm_yday  = time.tm_yday;
	result.tm_isdst = time.tm_isdst;
	return result;
}

static KYTY_SYSV_ABI GuestTm* libc_gmtime(const int64_t* timer) {
	if (timer == nullptr) {
		return nullptr;
	}

	thread_local GuestTm result {};
	std::tm              host_result {};
	const auto           t = static_cast<std::time_t>(*timer);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_gmtime64_s(&host_result, &t) != 0) {
		return nullptr;
	}
#else
	if (gmtime_r(&t, &host_result) == nullptr) {
		return nullptr;
	}
#endif

	result = ToGuestTm(host_result);
	return &result;
}

static KYTY_SYSV_ABI GuestTm* libc_localtime(const int64_t* timer) {
	if (timer == nullptr) {
		return nullptr;
	}

	thread_local GuestTm result {};
	std::tm              host_result {};
	const auto           t = static_cast<std::time_t>(*timer);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_localtime64_s(&host_result, &t) != 0) {
		return nullptr;
	}
#else
	if (localtime_r(&t, &host_result) == nullptr) {
		return nullptr;
	}
#endif

	result = ToGuestTm(host_result);
	return &result;
}

static KYTY_SYSV_ABI int64_t libc_mktime(GuestTm* timeptr) {
	if (timeptr == nullptr) {
		return -1;
	}

	auto host_time = ToHostTm(*timeptr);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto result = static_cast<int64_t>(_mktime64(&host_time));
#else
	const auto result = static_cast<int64_t>(std::mktime(&host_time));
#endif

	*timeptr = ToGuestTm(host_time);
	return result;
}

static KYTY_SYSV_ABI size_t libc_strftime(char* str, size_t count, const char* format,
                                          const GuestTm* timeptr) {
	if (str == nullptr || format == nullptr || timeptr == nullptr) {
		return 0;
	}

	const auto host_time = ToHostTm(*timeptr);
	return std::strftime(str, count, format, &host_time);
}

static KYTY_SYSV_ABI void catchReturnFromMain(int status) {
	PRINT_NAME();

	uint64_t ret_addr = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	uint64_t rsp      = 0;
	uint64_t rbp      = 0;
#if defined(__x86_64__) || defined(_M_X64)
	asm volatile("movq %%rsp, %0\n\t"
	             "movq %%rbp, %1\n\t"
	             : "=r"(rsp), "=r"(rbp));
#endif

	LOGF("\t status     = %d\n"
	     "\t ret_addr   = 0x%016" PRIx64 "\n"
	     "\t rsp        = 0x%016" PRIx64 "\n"
	     "\t rbp        = 0x%016" PRIx64 "\n",
	     status, ret_addr, rsp, rbp);
	if (rsp != 0) {
		auto* stack = reinterpret_cast<const uint64_t*>(rsp);
		for (uint32_t i = 0; i < 12; i++) {
			LOGF("\t stack[%02" PRIu32 "] = 0x%016" PRIx64 "\n", i, stack[i]);
		}
	}

	::printf("return from main = %d\n", status);
}

using execute_once_func_t = KYTY_SYSV_ABI int (*)(void*, void*, void**);

static std::mutex              g_execute_once_mutex;
static std::condition_variable g_execute_once_cv;

static KYTY_SYSV_ABI int std_execute_once(int* flag, execute_once_func_t func, void* arg) {
	PRINT_NAME();

	if (flag == nullptr || func == nullptr) {
		return 0;
	}

	constexpr int once_init    = 0;
	constexpr int once_done    = 1;
	constexpr int once_running = 2;

	{
		std::unique_lock lock(g_execute_once_mutex);
		while (*flag == once_running) {
			g_execute_once_cv.wait_for(lock, std::chrono::microseconds(10000));
			if (*flag == once_running) {
				lock.unlock();
				LibKernel::KernelDispatchPendingSignalForCurrentThread();
				lock.lock();
			}
		}
		if (*flag == once_done) {
			return 1;
		}
		*flag = once_running;
	}

	void* callback_context = nullptr;
	int   result           = func(nullptr, arg, &callback_context);

	{
		std::lock_guard lock(g_execute_once_mutex);
		*flag = (result != 0 ? once_done : once_init);
	}
	g_execute_once_cv.notify_all();

	return result;
}

static KYTY_SYSV_ABI int cxa_atexit(cxa_destructor_func_t func, void* arg, void* d) {
	PRINT_NAME();

	auto* cc = Common::Singleton<CContext>::Instance();

	CxaDestructor c {};
	c.destructor_func   = func;
	c.destructor_object = arg;
	c.module_id         = d;

	cc->cxa.push_back(c);

	return 0;
}

void KYTY_SYSV_ABI cxa_finalize(void* d) {
	PRINT_NAME();

	auto* cc = Common::Singleton<CContext>::Instance();

	for (auto i = cc->cxa.rbegin(); i != cc->cxa.rend(); ++i) {
		auto& c = *i;
		if ((d == nullptr || c.module_id == d) && c.destructor_func != nullptr) {
			auto* func        = c.destructor_func;
			auto* object      = c.destructor_object;
			c.destructor_func = nullptr;
			func(object);
		}
	}
}

static KYTY_SYSV_ABI int vsnprintf(char* s, size_t n, const char* format, VaList* va_list) {
	PRINT_NAME();

	// Preserve the caller's va_list state. Games commonly call vsnprintf
	// twice (first with a null buffer to measure, then to format), relying
	// on the va_list surviving the first call unchanged.
	VaList copy = *va_list;
	return GetGuestVsnprintfFunc()(s, n, format, &copy);
}

// Itanium C++ ABI __cxa_guard: guard object is 64 bits.
// Byte 0 = initialized flag; byte 1 = in-progress flag.
static std::mutex              g_cxa_guard_mutex;
static std::condition_variable g_cxa_guard_cv;

static KYTY_SYSV_ABI int cxa_guard_acquire(uint64_t* guard) {
	PRINT_NAME();

	auto* bytes = reinterpret_cast<uint8_t*>(guard);
	std::unique_lock lock(g_cxa_guard_mutex);
	g_cxa_guard_cv.wait(lock, [bytes] { return bytes[0] != 0 || bytes[1] == 0; });
	if (bytes[0] != 0) {
		return 0;
	}
	bytes[1] = 1;
	return 1;
}

static KYTY_SYSV_ABI void cxa_guard_release(uint64_t* guard) {
	PRINT_NAME();

	auto* bytes = reinterpret_cast<uint8_t*>(guard);
	{
		std::lock_guard lock(g_cxa_guard_mutex);
		bytes[0] = 1;
		bytes[1] = 0;
	}
	g_cxa_guard_cv.notify_all();
}

static KYTY_SYSV_ABI void cxa_guard_abort(uint64_t* guard) {
	PRINT_NAME();

	auto* bytes = reinterpret_cast<uint8_t*>(guard);
	{
		std::lock_guard lock(g_cxa_guard_mutex);
		bytes[1] = 0;
	}
	g_cxa_guard_cv.notify_all();
}

static KYTY_SYSV_ABI int bcmp(const void* a, const void* b, size_t n) {
	PRINT_NAME();
	return ::memcmp(a, b, n);
}

static KYTY_SYSV_ABI int rand() {
	// No PRINT_NAME: this is called in tight loops.
	return ::rand();
}

static KYTY_SYSV_ABI size_t libc_strspn(const char* s, const char* accept) {
	if (s == nullptr || accept == nullptr) {
		return 0;
	}
	return ::strspn(s, accept);
}

static KYTY_SYSV_ABI size_t libc_strcspn(const char* s, const char* reject) {
	if (s == nullptr || reject == nullptr) {
		return 0;
	}
	return ::strcspn(s, reject);
}

static KYTY_SYSV_ABI int strcasecmp(const char* a, const char* b) {
	if (a == nullptr || b == nullptr) {
		return (a == b) ? 0 : (a == nullptr ? -1 : 1);
	}
	while (*a != 0 && *b != 0) {
		int ca = std::tolower(static_cast<unsigned char>(*a));
		int cb = std::tolower(static_cast<unsigned char>(*b));
		if (ca != cb) {
			return ca - cb;
		}
		++a;
		++b;
	}
	return std::tolower(static_cast<unsigned char>(*a)) -
	       std::tolower(static_cast<unsigned char>(*b));
}

static KYTY_SYSV_ABI int fputs(const char* s, FILE* stream) {
	PRINT_NAME();
	if (s == nullptr) {
		return -1;
	}
	if (stream == stdout || stream == stderr || stream == nullptr) {
		LOGF_COLOR(Log::Color::BrightMagenta, "%s", s);
		return 0;
	}
	return ::fputs(s, stream);
}

// BSD-style thread-safe localtime: (const time_t* timer, struct tm* result).
// Sony's libc follows BSD; returns 0 on success.
static KYTY_SYSV_ABI int localtime_s(const int64_t* timer, GuestTm* result) {
	PRINT_NAME();
	if (timer == nullptr || result == nullptr) {
		return -1;
	}
	std::tm    host_result {};
	const auto t = static_cast<std::time_t>(*timer);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (_localtime64_s(&host_result, &t) != 0) {
		return -1;
	}
#else
	if (localtime_r(&t, &host_result) == nullptr) {
		return -1;
	}
#endif
	*result = ToGuestTm(host_result);
	return 0;
}

static KYTY_SYSV_ABI int strncasecmp(const char* a, const char* b, size_t n) {
	if (a == nullptr || b == nullptr || n == 0) {
		return 0;
	}
	for (size_t i = 0; i < n; ++i) {
		int ca = std::tolower(static_cast<unsigned char>(a[i]));
		int cb = std::tolower(static_cast<unsigned char>(b[i]));
		if (ca != cb || ca == 0) {
			return ca - cb;
		}
	}
	return 0;
}

static KYTY_SYSV_ABI char* libc_strcpy(char* dest, const char* src) {
	if (dest == nullptr || src == nullptr) {
		return dest;
	}
	return ::strcpy(dest, src);
}

static KYTY_SYSV_ABI float libc_logf(float x) {
	return ::logf(x);
}

static KYTY_SYSV_ABI double libc_atof(const char* s) {
	if (s == nullptr) {
		return 0.0;
	}
	return ::atof(s);
}

static KYTY_SYSV_ABI int64_t libc_clock() {
	// Microseconds since the first call; mirrors the CRT's monotonic clock().
	using clock_t = std::chrono::steady_clock;
	static const auto start = clock_t::now();
	auto               d     = clock_t::now() - start;
	return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

static KYTY_SYSV_ABI void* libc_memchr(const void* s, int c, size_t n) {
	if (s == nullptr || n == 0) {
		return nullptr;
	}
	return const_cast<void*>(::memchr(s, c, n));
}

// MSVC CRT internals used by the toupper / tolower macros in Sony's libc.
// Each returns a 256-entry unsigned short table mapping byte -> upper/lower.
// The game indexes directly: table[(unsigned char)c], so a null return
// crashes the scan at [table + c*2].
static const unsigned short* buildToUpperTable() {
	static unsigned short tbl[256];
	for (int i = 0; i < 256; ++i) {
		tbl[i] = static_cast<unsigned short>(std::toupper(i));
	}
	return tbl;
}
static const unsigned short* buildToLowerTable() {
	static unsigned short tbl[256];
	for (int i = 0; i < 256; ++i) {
		tbl[i] = static_cast<unsigned short>(std::tolower(i));
	}
	return tbl;
}
static KYTY_SYSV_ABI const unsigned short* Getptoupper() {
	static const unsigned short* p = buildToUpperTable();
	return p;
}
static KYTY_SYSV_ABI const unsigned short* Getptolower() {
	static const unsigned short* p = buildToLowerTable();
	return p;
}

static KYTY_SYSV_ABI char* libc_strncat(char* dest, const char* src, size_t n) {
	if (dest == nullptr || src == nullptr) {
		return dest;
	}
	return ::strncat(dest, src, n);
}

// Guest allocator pass-through: route to host CRT. Guest-side free (NID
// tIhsqj0qsFE) is wired to ::free so the pair stays consistent. The game's
// own operator new/delete may still use its own allocator; the only safety
// requirement here is "whatever libc_v1 malloc returned, libc_v1 free gets".
static KYTY_SYSV_ABI void* libc_malloc(size_t n) {
	return ::malloc(n);
}
static KYTY_SYSV_ABI void libc_free(void* p) {
	::free(p);
}
static KYTY_SYSV_ABI void* libc_calloc(size_t n, size_t sz) {
	return ::calloc(n, sz);
}
static KYTY_SYSV_ABI void* libc_realloc(void* p, size_t n) {
	return ::realloc(p, n);
}

// PS5 wchar_t is 16-bit (UTF-16 code unit). Host wchar_t is 32-bit on Linux
// and MSVC-compatible 16-bit on Windows; use char16_t directly to stay
// portable regardless of host.
static KYTY_SYSV_ABI size_t libc_wcslen(const char16_t* s) {
	if (s == nullptr) {
		return 0;
	}
	size_t n = 0;
	while (s[n] != 0) {
		++n;
	}
	return n;
}

static KYTY_SYSV_ABI int libc_wcsncmp(const char16_t* a, const char16_t* b, size_t n) {
	if (a == nullptr || b == nullptr || n == 0) {
		return 0;
	}
	for (size_t i = 0; i < n; ++i) {
		if (a[i] != b[i]) {
			return (a[i] < b[i]) ? -1 : 1;
		}
		if (a[i] == 0) {
			return 0;
		}
	}
	return 0;
}

static KYTY_SYSV_ABI int libc_wmemcmp(const char16_t* a, const char16_t* b, size_t n) {
	if (a == nullptr || b == nullptr || n == 0) {
		return 0;
	}
	for (size_t i = 0; i < n; ++i) {
		if (a[i] != b[i]) {
			return (a[i] < b[i]) ? -1 : 1;
		}
	}
	return 0;
}

static KYTY_SYSV_ABI char16_t* libc_wcscpy(char16_t* dest, const char16_t* src) {
	if (dest == nullptr || src == nullptr) {
		return dest;
	}
	char16_t* d = dest;
	while ((*d++ = *src++) != 0) {
	}
	return dest;
}

static KYTY_SYSV_ABI int libc_wcscmp(const char16_t* a, const char16_t* b) {
	if (a == nullptr || b == nullptr) {
		return (a == b) ? 0 : (a == nullptr ? -1 : 1);
	}
	while (*a != 0 && *a == *b) {
		++a;
		++b;
	}
	if (*a == *b) {
		return 0;
	}
	return (*a < *b) ? -1 : 1;
}

// MSVC STL internal: find first element in [first,last) equal to value, where
// element size matches the suffix (1/2/4/8 bytes). Returns `last` if no hit.
static KYTY_SYSV_ABI const void* std_find_trivial_1(const void* first, const void* last, uint8_t v) {
	auto* p   = static_cast<const uint8_t*>(first);
	auto* end = static_cast<const uint8_t*>(last);
	for (; p < end; ++p) {
		if (*p == v) {
			return p;
		}
	}
	return end;
}
static KYTY_SYSV_ABI const void* std_find_trivial_2(const void* first, const void* last, uint16_t v) {
	auto* p   = static_cast<const uint16_t*>(first);
	auto* end = static_cast<const uint16_t*>(last);
	for (; p < end; ++p) {
		if (*p == v) {
			return p;
		}
	}
	return end;
}
static KYTY_SYSV_ABI const void* std_find_trivial_4(const void* first, const void* last, uint32_t v) {
	auto* p   = static_cast<const uint32_t*>(first);
	auto* end = static_cast<const uint32_t*>(last);
	for (; p < end; ++p) {
		if (*p == v) {
			return p;
		}
	}
	return end;
}
static KYTY_SYSV_ABI const void* std_find_trivial_8(const void* first, const void* last, uint64_t v) {
	auto* p   = static_cast<const uint64_t*>(first);
	auto* end = static_cast<const uint64_t*>(last);
	for (; p < end; ++p) {
		if (*p == v) {
			return p;
		}
	}
	return end;
}

// Lazy setjmp/longjmp: PS5 ships setjmp inline-asm thunks that we cannot
// faithfully re-create without owning the guest's register save area. Return
// 0 on every call so the "first time through" path runs; the game's early
// boot has yet to invoke longjmp, so leaving that half unwired is safe for
// now. Revisit when a matching longjmp NID appears in the log.
static KYTY_SYSV_ABI int libc_setjmp(void* /*env*/) {
	return 0;
}

static KYTY_SYSV_ABI float libc_expf(float x) {
	return ::expf(x);
}
static KYTY_SYSV_ABI float libc_tanf(float x) {
	return ::tanf(x);
}
static KYTY_SYSV_ABI float libc_log2f(float x) {
	return ::log2f(x);
}
static KYTY_SYSV_ABI float libc_powf(float x, float y) {
	return ::powf(x, y);
}

// Microsoft-style bounds-checked strncpy_s. Matches the Annex K signature
// used by Sony's libc: writes up to `count` bytes (plus NUL) into `dest`,
// returning 0 on success or an errno on failure. Treat zero-sized or null
// args defensively to avoid cascading faults.
static KYTY_SYSV_ABI int libc_strncpy_s(char* dest, size_t dest_size,
                                        const char* src, size_t count) {
	if (dest == nullptr || dest_size == 0) {
		return 22; // EINVAL
	}
	if (src == nullptr) {
		dest[0] = '\0';
		return 22;
	}
	size_t src_len = ::strnlen(src, count);
	if (src_len >= dest_size) {
		dest[0] = '\0';
		return 34; // ERANGE
	}
	::memcpy(dest, src, src_len);
	dest[src_len] = '\0';
	return 0;
}

static KYTY_SYSV_ABI const char16_t* libc_wcschr(const char16_t* s, char16_t c) {
	if (s == nullptr) {
		return nullptr;
	}
	while (*s != 0) {
		if (*s == c) {
			return s;
		}
		++s;
	}
	return (c == 0) ? s : nullptr;
}

static KYTY_SYSV_ABI const char16_t* libc_wmemchr(const char16_t* s, char16_t c, size_t n) {
	if (s == nullptr || n == 0) {
		return nullptr;
	}
	for (size_t i = 0; i < n; ++i) {
		if (s[i] == c) {
			return s + i;
		}
	}
	return nullptr;
}

static KYTY_SYSV_ABI char16_t* libc_wmemset(char16_t* dest, char16_t c, size_t n) {
	if (dest == nullptr) {
		return dest;
	}
	for (size_t i = 0; i < n; ++i) {
		dest[i] = c;
	}
	return dest;
}

// wcstol / wcstof: narrow the UTF-16 input to ASCII (digits + sign + dot
// + E/e) and defer to the host CRT. The games we care about pass numeric
// strings that fit entirely in ASCII, which keeps this transliteration safe.
static std::string widenAsciiToString(const char16_t* w, const char16_t** end_out) {
	std::string out;
	if (w == nullptr) {
		if (end_out != nullptr) {
			*end_out = nullptr;
		}
		return out;
	}
	const char16_t* p = w;
	while (*p != 0) {
		auto v = static_cast<uint32_t>(*p);
		if (v > 0x7f) {
			break;
		}
		out.push_back(static_cast<char>(v));
		++p;
	}
	if (end_out != nullptr) {
		*end_out = p;
	}
	return out;
}

static KYTY_SYSV_ABI int64_t libc_wcstol(const char16_t* s, char16_t** endptr, int base) {
	const char16_t* w_end = nullptr;
	std::string     narrow = widenAsciiToString(s, &w_end);
	char*           n_end  = nullptr;
	int64_t         v      = ::strtoll(narrow.c_str(), &n_end, base);
	if (endptr != nullptr) {
		const char* n_begin = narrow.c_str();
		size_t      offset  = static_cast<size_t>(n_end - n_begin);
		*endptr             = const_cast<char16_t*>(s + offset);
	}
	return v;
}

static KYTY_SYSV_ABI float libc_wcstof(const char16_t* s, char16_t** endptr) {
	const char16_t* w_end = nullptr;
	std::string     narrow = widenAsciiToString(s, &w_end);
	char*           n_end  = nullptr;
	float           v      = ::strtof(narrow.c_str(), &n_end);
	if (endptr != nullptr) {
		const char* n_begin = narrow.c_str();
		size_t      offset  = static_cast<size_t>(n_end - n_begin);
		*endptr             = const_cast<char16_t*>(s + offset);
	}
	return v;
}

// Itanium ABI sized operator delete: _ZdlPvm (operator delete(void*, size_t)).
// The game owns its allocator (see MallocReplace in libKernel.cpp); any ptr
// here was handed back by that guest allocator, not host ::malloc, so freeing
// it with the host CRT corrupts the host heap. Treat as a leak until the
// game's own _ZdlPv (NID z+P+xCnWLBk) is routed back here.
static KYTY_SYSV_ABI void sized_operator_delete(void* /*ptr*/, size_t /*size*/) {
	PRINT_NAME();
}

} // namespace LibC

namespace LibcInternalExt {

LIB_VERSION("LibcInternalExt", 1, "LibcInternal", 1, 1);

static uint64_t g_mspace_atomic_id_mask = 0;
static uint64_t g_mstate_table[64]      = {0};

using thread_atexit_destructor_t = KYTY_SYSV_ABI void (*)(void*);

struct ThreadAtexitDestructor {
	thread_atexit_destructor_t destructor;
	void*                      object;
};

static thread_local std::vector<ThreadAtexitDestructor> g_thread_atexit_destructors;

struct Info {
	uint64_t  size;
	uint32_t  unknown1;
	uint32_t  unknown2;
	uint64_t* mspace_atomic_id_mask;
	uint64_t* mstate_table;
};

void KYTY_SYSV_ABI LibcHeapGetTraceInfo(Info* info) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(info->size != 32);

	info->mspace_atomic_id_mask = &g_mspace_atomic_id_mask;
	info->mstate_table          = g_mstate_table;
}

int KYTY_SYSV_ABI LibcInternalExtCxaThreadAtexit(thread_atexit_destructor_t destructor,
                                                 void* object, void* /*module_id*/) {
	PRINT_NAME();

	g_thread_atexit_destructors.push_back({destructor, object});

	return 0;
}

void RunThreadAtexitDestructors() {
	while (!g_thread_atexit_destructors.empty()) {
		auto destructor = g_thread_atexit_destructors.back();
		g_thread_atexit_destructors.pop_back();

		if (destructor.destructor != nullptr) {
			destructor.destructor(destructor.object);
		}
	}
}

LIB_DEFINE(InitLibcInternalExt_1) {
	LIB_FUNC("NWtTN10cJzE", LibcInternalExt::LibcHeapGetTraceInfo);
	LIB_FUNC("qBS714-Jr3g", LibcInternalExt::LibcInternalExtCxaThreadAtexit);
}

} // namespace LibcInternalExt

namespace LibcInternal {

LIB_VERSION("LibcInternal", 1, "LibcInternal", 1, 1);

static uint32_t g_need_flag = 1;

int KYTY_SYSV_ABI vprintf(const char* str, VaList* c) {
	PRINT_NAME();

	return GetGuestVprintfFunc()(str, c);
}

static KYTY_SYSV_ABI int snprintf(VA_ARGS) {
	VA_CONTEXT(ctx); // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

	PRINT_NAME();

	return GetGuestSnprintfCtxFunc()(&ctx);
}

int KYTY_SYSV_ABI fflush(FILE* stream) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(stream != stdout);

	return ::fflush(stream);
}

void* KYTY_SYSV_ABI memset(void* s, int c, size_t n) {
	PRINT_NAME();

	return ::memset(s, c, n);
}

void* KYTY_SYSV_ABI memcpy(void* dest, const void* src, size_t n) {
	return ::memcpy(dest, src, n);
}

void* KYTY_SYSV_ABI memmove(void* dest, const void* src, size_t n) {
	return ::memmove(dest, src, n);
}

int KYTY_SYSV_ABI memcmp(const void* s1, const void* s2, size_t n) {
	return ::memcmp(s1, s2, n);
}

int KYTY_SYSV_ABI strcmp(const char* s1, const char* s2) {
	return ::strcmp(s1, s2);
}

int KYTY_SYSV_ABI strncmp(const char* s1, const char* s2, size_t n) {
	return ::strncmp(s1, s2, n);
}

size_t KYTY_SYSV_ABI strlen(const char* s) {
	return ::strlen(s);
}

char* KYTY_SYSV_ABI strcpy(char* dest, const char* src) {
	return ::strcpy(dest, src);
}

char* KYTY_SYSV_ABI strncpy(char* dest, const char* src, size_t count) {
	return ::strncpy(dest, src, count);
}

char* KYTY_SYSV_ABI strcat(char* dest, const char* src) {
	return ::strcat(dest, src);
}

const char* KYTY_SYSV_ABI strchr(const char* s, int c) {
	return ::strchr(s, c);
}

char* KYTY_SYSV_ABI strrchr(const char* s, int c) {
	return const_cast<char*>(::strrchr(s, c));
}

char* KYTY_SYSV_ABI strstr(const char* haystack, const char* needle) {
	static int log_count = 0;
	if (log_count++ < 8) {
		LOGF("LibcInternal::strstr(\"%s\", \"%s\")\n", (haystack != nullptr ? haystack : "<null>"),
		     (needle != nullptr ? needle : "<null>"));
	}
	return const_cast<char*>(::strstr(haystack, needle));
}

long KYTY_SYSV_ABI strtol(const char* str, char** endptr, int base) {
	static int log_count = 0;
	if (log_count++ < 16) {
		LOGF("LibcInternal::strtol(\"%s\", base=%d)\n", (str != nullptr ? str : "<null>"), base);
	}
	return ::strtol(str, endptr, base);
}

unsigned long KYTY_SYSV_ABI strtoul(const char* str, char** endptr, int base) {
	static int log_count = 0;
	if (log_count++ < 16) {
		LOGF("LibcInternal::strtoul(\"%s\", base=%d)\n", (str != nullptr ? str : "<null>"), base);
	}
	return ::strtoul(str, endptr, base);
}

int KYTY_SYSV_ABI atoi(const char* str) {
	return ::atoi(str);
}

float KYTY_SYSV_ABI sinf(float x) {
	return std::sin(x);
}

float KYTY_SYSV_ABI cosf(float x) {
	return std::cos(x);
}

void KYTY_SYSV_ABI sincosf(float x, float* sinp, float* cosp) {
	if (sinp != nullptr) {
		*sinp = std::sin(x);
	}
	if (cosp != nullptr) {
		*cosp = std::cos(x);
	}
}

double KYTY_SYSV_ABI sin(double x) {
	return std::sin(x);
}

double KYTY_SYSV_ABI cos(double x) {
	return std::cos(x);
}

void KYTY_SYSV_ABI sincos(double x, double* sinp, double* cosp) {
	if (sinp != nullptr) {
		*sinp = std::sin(x);
	}
	if (cosp != nullptr) {
		*cosp = std::cos(x);
	}
}

int KYTY_SYSV_ABI LibcHeapErrorReportForGame(uint64_t msp, uint64_t ptr, uint64_t error,
                                             uint64_t arg3, uint64_t arg4, uint64_t arg5) {
	PRINT_NAME();

	LOGF("\t temporary: heap error report ignored, msp=0x%016" PRIx64 ", ptr=0x%016" PRIx64
	     ", error=0x%016" PRIx64 ", args=(0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 ")\n",
	     msp, ptr, error, arg3, arg4, arg5);

	return 0;
}

LIB_DEFINE(InitLibcInternal_1) {
	LibcInternalExt::InitLibcInternalExt_1(s);

	LIB_OBJECT("ZT4ODD2Ts9o", &LibcInternal::g_need_flag);
	LIB_OBJECT("2sWzhYqFH4E", stdout);

	LIB_FUNC("GMpvxPFW924", LibcInternal::vprintf);
	LIB_FUNC("MUjC4lbHrK4", LibcInternal::fflush);
	LIB_FUNC("8zTFvBIAIN8", LibcInternal::memset);
	LIB_FUNC("Q3VBxCXhUHs", LibcInternal::memcpy);
	LIB_FUNC("+P6FRGH4LfA", LibcInternal::memmove);
	LIB_FUNC("DfivPArhucg", LibcInternal::memcmp);
	LIB_FUNC("aesyjrHVWy4", LibcInternal::strcmp);
	LIB_FUNC("Ovb2dSJOAuE", LibcInternal::strncmp);
	LIB_FUNC("j4ViWNHEgww", LibcInternal::strlen);
	LIB_FUNC("5Xa2ACNECdo", LibcInternal::strcpy);
	LIB_FUNC("6sJWiWSRuqk", LibcInternal::strncpy);
	LIB_FUNC("Ls4tzzhimqQ", LibcInternal::strcat);
	LIB_FUNC("ob5xAW4ln-0", LibcInternal::strchr);
	LIB_FUNC("9yDWMxEFdJU", LibcInternal::strrchr);
	LIB_FUNC("viiwFMaNamA", LibcInternal::strstr);
	LIB_FUNC("mXlxhmLNMPg", LibcInternal::strtol);
	LIB_FUNC("QxmSHBCuKTk", LibcInternal::strtoul);
	LIB_FUNC("zlfEH8FmyUA", LibcInternal::strtoul);
	LIB_FUNC("fPxypibz2MY", LibcInternal::atoi);
	LIB_FUNC("Q4rRL34CEeE", LibcInternal::sinf);
	LIB_FUNC("-P6FNMzk2Kc", LibcInternal::cosf);
	LIB_FUNC("pztV4AF18iI", LibcInternal::sincosf);
	LIB_FUNC("H8ya2H00jbI", LibcInternal::sin);
	LIB_FUNC("2WE3BTYVwKM", LibcInternal::cos);
	LIB_FUNC("jMB7EFyu30Y", LibcInternal::sincos);
	LIB_FUNC("eLdDw6l0-bU", LibcInternal::snprintf);

	LIB_FUNC("L1SBTkC+Cvw", LibC::abort);
	LIB_FUNC("tsvEmnenz48", LibC::cxa_atexit);
	LIB_FUNC("H2e8t5ScQGc", LibC::cxa_finalize);
	LIB_FUNC("DiGVep5yB5w", LibC::std_execute_once);

	LIB_FUNC("al3JzFI9MQ0", LibcInternal::LibcHeapErrorReportForGame);
}

} // namespace LibcInternal

namespace LibC {

LIB_DEFINE(InitLibC_1) {
	LibcInternal::InitLibcInternal_1(s);

	LIB_OBJECT("P330P3dFF68", &LibC::g_need_flag);

	LIB_FUNC("uMei1W9uyNo", LibC::exit);
	LIB_FUNC("L1SBTkC+Cvw", LibC::abort);
	LIB_FUNC("9BcDykPmo1I", LibC::libc_error);
	LIB_FUNC("bzQExy189ZI", LibC::init_env);
	LIB_FUNC("8G2LB+A3rzg", LibC::atexit);
	LIB_FUNC("hcuQgD53UxM", LibC::libc_printf);
	LIB_FUNC("YQ0navp+YIc", LibC::puts);
	LIB_FUNC("M4YYbSFfJ8g", LibC::setenv);
	LIB_FUNC("wLlFkwG9UcQ", LibC::libc_time);
	LIB_FUNC("-VVn74ZyhEs", LibC::libc_difftime);
	LIB_FUNC("1mecP7RgI2A", LibC::libc_gmtime);
	LIB_FUNC("efhK-YSUYYQ", LibC::libc_localtime);
	LIB_FUNC("n7AepwR0s34", LibC::libc_mktime);
	LIB_FUNC("Av3zjWi64Kw", LibC::libc_strftime);
	LIB_FUNC("XKRegsFpEpk", LibC::catchReturnFromMain);
	LIB_FUNC("tsvEmnenz48", LibC::cxa_atexit);
	LIB_FUNC("H2e8t5ScQGc", LibC::cxa_finalize);
	LIB_FUNC("DiGVep5yB5w", LibC::std_execute_once);
	LIB_FUNC("Q2V+iqvjgC0", LibC::vsnprintf);
	LIB_FUNC("3GPpjQdAMTw", LibC::cxa_guard_acquire);
	LIB_FUNC("9rAeANT2tyE", LibC::cxa_guard_release);
	LIB_FUNC("2emaaluWzUw", LibC::cxa_guard_abort);
	LIB_FUNC("5TjaJwkLWxE", LibC::bcmp);
	LIB_FUNC("cpCOXWMgha0", LibC::rand);
	LIB_FUNC("-kU6bB4M-+k", LibC::libc_strspn);
	LIB_FUNC("q0F6yS-rCms", LibC::libc_strcspn);
	LIB_FUNC("AV6ipCNa4Rw", LibC::strcasecmp);
	LIB_FUNC("QrZZdJ8XsX0", LibC::fputs);
	LIB_FUNC("fiiNDnNBKVY", LibC::localtime_s);
	LIB_FUNC("lYDzBVE5mZs", LibC::sized_operator_delete);
	LIB_FUNC("pXvbDfchu6k", LibC::strncasecmp);
	LIB_FUNC("kiZSXIWd9vg", LibC::libc_strcpy);
	LIB_FUNC("RQXLbdT2lc4", LibC::libc_logf);
	LIB_FUNC("SRI6S9B+-a4", LibC::libc_atof);
	LIB_FUNC("QZP6I9ZZxpE", LibC::libc_clock);
	LIB_FUNC("8u8lPzUEq+U", LibC::libc_memchr);
	LIB_FUNC("rcQCUr0EaRU", LibC::Getptoupper);
	LIB_FUNC("1uJgoVq3bQU", LibC::Getptolower);
	LIB_FUNC("kHg45qPC6f0", LibC::libc_strncat);
	LIB_FUNC("gQX+4GDQjpM", LibC::libc_malloc);
	LIB_FUNC("tIhsqj0qsFE", LibC::libc_free);
	LIB_FUNC("2X5agFjKxMc", LibC::libc_calloc);
	LIB_FUNC("Y7aJ1uydPMo", LibC::libc_realloc);
	LIB_FUNC("WkkeywLJcgU", LibC::libc_wcslen);
	LIB_FUNC("E8wCoUEbfzk", LibC::libc_wcsncmp);
	LIB_FUNC("QJ5xVfKkni0", LibC::libc_wmemcmp);
	LIB_FUNC("Al8MZJh-4hM", LibC::libc_wmemset);
	LIB_FUNC("d3dMyWORw8A", LibC::libc_wcstol);
	LIB_FUNC("7SXNu+0KBYQ", LibC::libc_wcstof);
	LIB_FUNC("Ezzq78ZgHPs", LibC::libc_wcschr);
	LIB_FUNC("fnUEjBCNRVU", LibC::libc_wmemchr);
	LIB_FUNC("FM5NPnLqBc8", LibC::libc_wcscpy);
	LIB_FUNC("pNtJdE3x49E", LibC::libc_wcscmp);
	LIB_FUNC("aRtxlukU9WQ", LibC::std_find_trivial_1);
	LIB_FUNC("Oa7AB0wiixE", LibC::std_find_trivial_2);
	LIB_FUNC("Zf7ewes1VtY", LibC::std_find_trivial_4);
	LIB_FUNC("3sssb+7Wn9M", LibC::std_find_trivial_8);
	LIB_FUNC("gNQ1V2vfXDE", LibC::libc_setjmp);
	LIB_FUNC("8zsu04XNsZ4", LibC::libc_expf);
	LIB_FUNC("ZE6RNL+eLbk", LibC::libc_tanf);
	LIB_FUNC("hsi9drzHR2k", LibC::libc_log2f);
	LIB_FUNC("1D0H2KNjshE", LibC::libc_powf);
	LIB_FUNC("YNzNkJzYqEg", LibC::libc_strncpy_s);
}

} // namespace LibC

} // namespace Libs
