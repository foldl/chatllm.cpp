OUTPUT_PATH=obj
GGML_BASE=third_party/ggml
GGML_SRC=$(GGML_BASE)/src
GGML_INC=$(GGML_BASE)/include/ggml

# Define the default target now so that it is always the first target
BUILD_TARGETS = $(OUTPUT_PATH)/main

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifndef CHATLLM_NO_METAL
		CHATLLM_METAL := 1
	endif

	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

default: $(OUTPUT_PATH)/ $(BUILD_TARGETS)

all: $(BUILD_TARGETS)

$(OUTPUT_PATH)/:
	mkdir $(OUTPUT_PATH)

ifdef RISCV_CROSS_COMPILE
CC	:= riscv64-unknown-linux-gnu-gcc
CXX	:= riscv64-unknown-linux-gnu-g++
endif

#
# Compile flags
#

# keep standard at C11 and C++20
MK_CPPFLAGS  = -I. -I$(GGML_BASE) -I$(GGML_INC) -I$(GGML_SRC)
MK_CFLAGS    = -std=c11   -fPIC
MK_CXXFLAGS  = -std=c++20 -fPIC
MK_NVCCFLAGS = -std=c++20

# -Ofast tends to produce faster code, but may not be available for some compilers.
ifdef CHATLLM_FAST
MK_CFLAGS     += -Ofast
HOST_CXXFLAGS += -Ofast
MK_NVCCFLAGS  += -O3
else
MK_CFLAGS     += -O3
MK_CXXFLAGS   += -O3
MK_NVCCFLAGS  += -O3
endif

ifndef CHATLLM_NO_CCACHE
CCACHE := $(shell which ccache)
ifdef CCACHE
export CCACHE_SLOPPINESS = time_macros
$(info I ccache found, compilation results will be cached. Disable with CHATLLM_NO_CCACHE.)
CC    := $(CCACHE) $(CC)
CXX   := $(CCACHE) $(CXX)
else
$(info I ccache not found. Consider installing it for faster compilation.)
endif # CCACHE
endif # CHATLLM_NO_CCACHE

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
MK_CPPFLAGS += -D_XOPEN_SOURCE=600

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
ifeq ($(UNAME_S),OpenBSD)
	MK_CPPFLAGS += -U_XOPEN_SOURCE -D_XOPEN_SOURCE=700
endif

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
ifeq ($(UNAME_S),Linux)
	MK_CPPFLAGS += -D_GNU_SOURCE
endif

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
ifeq ($(UNAME_S),Darwin)
	MK_CPPFLAGS += -D_DARWIN_C_SOURCE
endif
ifeq ($(UNAME_S),DragonFly)
	MK_CPPFLAGS += -D__BSD_VISIBLE
endif

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
ifeq ($(UNAME_S),FreeBSD)
	MK_CPPFLAGS += -D__BSD_VISIBLE
endif
ifeq ($(UNAME_S),NetBSD)
	MK_CPPFLAGS += -D_NETBSD_SOURCE
endif
ifeq ($(UNAME_S),OpenBSD)
	MK_CPPFLAGS += -D_BSD_SOURCE
endif

ifdef CHATLLM_SCHED_MAX_COPIES
	MK_CPPFLAGS += -DGGML_SCHED_MAX_COPIES=$(CHATLLM_SCHED_MAX_COPIES)
endif

ifdef CHATLLM_DEBUG
	MK_CFLAGS   += -O0 -g
	MK_CXXFLAGS += -O0 -g
	MK_LDFLAGS  += -g

	ifeq ($(UNAME_S),Linux)
		MK_CPPFLAGS += -D_GLIBCXX_ASSERTIONS
	endif
else
	MK_CPPFLAGS += -DNDEBUG
endif

ifdef CHATLLM_SANITIZE_THREAD
	MK_CFLAGS   += -fsanitize=thread -g
	MK_CXXFLAGS += -fsanitize=thread -g
	MK_LDFLAGS  += -fsanitize=thread -g
endif

ifdef CHATLLM_SANITIZE_ADDRESS
	MK_CFLAGS   += -fsanitize=address -fno-omit-frame-pointer -g
	MK_CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
	MK_LDFLAGS  += -fsanitize=address -fno-omit-frame-pointer -g
endif

ifdef CHATLLM_SANITIZE_UNDEFINED
	MK_CFLAGS   += -fsanitize=undefined -g
	MK_CXXFLAGS += -fsanitize=undefined -g
	MK_LDFLAGS  += -fsanitize=undefined -g
endif

ifdef CHATLLM_SERVER_VERBOSE
	MK_CPPFLAGS += -DSERVER_VERBOSE=$(CHATLLM_SERVER_VERBOSE)
endif

ifdef CHATLLM_SERVER_SSL
	MK_CPPFLAGS += -DCPPHTTPLIB_OPENSSL_SUPPORT
	MK_LDFLAGS += -lssl -lcrypto
endif

ifdef CHATLLM_DISABLE_LOGS
	MK_CPPFLAGS += -DLOG_DISABLE_LOGS
endif # CHATLLM_DISABLE_LOGS

# warnings
WARN_FLAGS    = -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-unused-parameter
MK_CFLAGS    += $(WARN_FLAGS) -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int \
				-Werror=implicit-function-declaration
MK_CXXFLAGS  += $(WARN_FLAGS) -Wmissing-declarations -Wmissing-noreturn

ifeq ($(CHATLLM_FATAL_WARNINGS),1)
	MK_CFLAGS   += -Werror
	MK_CXXFLAGS += -Werror
endif

# this version of Apple ld64 is buggy
ifneq '' '$(findstring dyld-1015.7,$(shell $(CC) $(LDFLAGS) -Wl,-v 2>&1))'
	MK_CPPFLAGS += -DHAVE_BUGGY_APPLE_LINKER
endif

# OS specific
# TODO: support Windows
ifneq '' '$(filter $(UNAME_S),Linux Darwin FreeBSD NetBSD OpenBSD Haiku)'
	MK_CFLAGS   += -pthread
	MK_CXXFLAGS += -pthread
endif

# detect Windows
ifneq ($(findstring _NT,$(UNAME_S)),)
	_WIN32 := 1
endif

# library name prefix
ifneq ($(_WIN32),1)
	LIB_PRE := lib
endif

# Dynamic Shared Object extension
ifneq ($(_WIN32),1)
	DSO_EXT := .so
else
	DSO_EXT := .dll
endif

# Windows Sockets 2 (Winsock) for network-capable apps
ifeq ($(_WIN32),1)
	LWINSOCK2 := -lws2_32
	MK_LDFLAGS := -municode
endif

ifdef CHATLLM_GPROF
	MK_CFLAGS   += -pg
	MK_CXXFLAGS += -pg
endif
ifdef CHATLLM_PERF
	MK_CPPFLAGS += -DGGML_PERF
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue

ifndef RISCV

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	MK_CFLAGS     += -march=native -mtune=native
	HOST_CXXFLAGS += -march=native -mtune=native

	# Usage AVX-only
	#MK_CFLAGS   += -mfma -mf16c -mavx
	#MK_CXXFLAGS += -mfma -mf16c -mavx

	# Usage SSSE3-only (Not is SSE3!)
	#MK_CFLAGS   += -mssse3
	#MK_CXXFLAGS += -mssse3
endif

ifneq '' '$(findstring mingw,$(shell $(CC) -dumpmachine))'
	# The stack is only 16-byte aligned on Windows, so don't let gcc emit aligned moves.
	# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412
	# https://github.com/ggerganov/llama.cpp/issues/2922
	MK_CFLAGS   += -Xassembler -muse-unaligned-vector-move
	MK_CXXFLAGS += -Xassembler -muse-unaligned-vector-move

	# Target Windows 8 for PrefetchVirtualMemory
	MK_CPPFLAGS += -D_WIN32_WINNT=0x602
endif

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	# Nvidia Jetson
	MK_CFLAGS   += -mcpu=native
	MK_CXXFLAGS += -mcpu=native
	JETSON_RELEASE_INFO = $(shell jetson_release)
	ifdef JETSON_RELEASE_INFO
		ifneq ($(filter TX2%,$(JETSON_RELEASE_INFO)),)
			JETSON_EOL_MODULE_DETECT = 1
			CC = aarch64-unknown-linux-gnu-gcc
			cxx = aarch64-unknown-linux-gnu-g++
		endif
	endif
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif

ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	MK_CFLAGS   += -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		MK_CFLAGS   += -mcpu=power9
		MK_CXXFLAGS += -mcpu=power9
	endif
endif

ifneq ($(filter ppc64le%,$(UNAME_M)),)
	MK_CFLAGS   += -mcpu=powerpc64le
	MK_CXXFLAGS += -mcpu=powerpc64le
	CUDA_POWER_ARCH = 1
endif

else
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

ifdef CHATLLM_QKK_64
	MK_CPPFLAGS += -DGGML_QKK_64
endif

ifndef CHATLLM_NO_ACCELERATE
	# Mac OS - include Accelerate framework.
	# `-framework Accelerate` works both with Apple Silicon and Mac Intel
	ifeq ($(UNAME_S),Darwin)
		MK_CPPFLAGS += -DGGML_USE_ACCELERATE
		MK_CPPFLAGS += -DACCELERATE_NEW_LAPACK
		MK_CPPFLAGS += -DACCELERATE_LAPACK_ILP64
		MK_LDFLAGS  += -framework Accelerate
	endif
endif # CHATLLM_NO_ACCELERATE

ifdef CHATLLM_MPI
	MK_CPPFLAGS += -DGGML_USE_MPI
	MK_CFLAGS   += -Wno-cast-qual
	MK_CXXFLAGS += -Wno-cast-qual
	OBJS        += $(OUTPUT_PATH)/ggml-mpi.o
endif # CHATLLM_MPI

ifdef CHATLLM_OPENBLAS
	MK_CPPFLAGS += -DGGML_USE_OPENBLAS $(shell pkg-config --cflags-only-I openblas)
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other openblas)
	MK_LDFLAGS  += $(shell pkg-config --libs openblas)
endif # CHATLLM_OPENBLAS

ifdef CHATLLM_BLIS
	MK_CPPFLAGS += -DGGML_USE_OPENBLAS -I/usr/local/include/blis -I/usr/include/blis
	MK_LDFLAGS  += -lblis -L/usr/local/lib
endif # CHATLLM_BLIS

ifdef CHATLLM_CUBLAS
# CHATLLM_CUBLAS is deprecated and will be removed in the future
	CHATLLM_CUDA := 1
endif

ifdef CHATLLM_CUDA
	ifneq ('', '$(wildcard /opt/cuda)')
		CUDA_PATH ?= /opt/cuda
	else
		CUDA_PATH ?= /usr/local/cuda
	endif
	MK_CPPFLAGS  += -DGGML_USE_CUDA -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/$(UNAME_M)-linux/include
	MK_LDFLAGS   += -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L$(CUDA_PATH)/lib64 -L/usr/lib64 -L$(CUDA_PATH)/targets/$(UNAME_M)-linux/lib -L/usr/lib/wsl/lib
	OBJS         += ggml-cuda.o
	OBJS         += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/*.cu))
	MK_NVCCFLAGS += -use_fast_math
ifdef CHATLLM_FATAL_WARNINGS
	MK_NVCCFLAGS += -Werror all-warnings
endif # CHATLLM_FATAL_WARNINGS
ifndef JETSON_EOL_MODULE_DETECT
	MK_NVCCFLAGS += --forward-unknown-to-host-compiler
endif # JETSON_EOL_MODULE_DETECT
ifdef CHATLLM_DEBUG
	MK_NVCCFLAGS += -lineinfo
endif # CHATLLM_DEBUG
ifdef CHATLLM_CUDA_NVCC
	NVCC = $(CCACHE) $(CHATLLM_CUDA_NVCC)
else
	NVCC = $(CCACHE) nvcc
endif #CHATLLM_CUDA_NVCC
ifdef CUDA_DOCKER_ARCH
	MK_NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else ifndef CUDA_POWER_ARCH
	MK_NVCCFLAGS += -arch=native
endif # CUDA_DOCKER_ARCH
ifdef CHATLLM_CUDA_FORCE_DMMV
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_DMMV
endif # CHATLLM_CUDA_FORCE_DMMV
ifdef CHATLLM_CUDA_FORCE_MMQ
	MK_NVCCFLAGS += -DGGML_CUDA_FORCE_MMQ
endif # CHATLLM_CUDA_FORCE_MMQ
ifdef CHATLLM_CUDA_DMMV_X
	MK_NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(CHATLLM_CUDA_DMMV_X)
else
	MK_NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # CHATLLM_CUDA_DMMV_X
ifdef CHATLLM_CUDA_MMV_Y
	MK_NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(CHATLLM_CUDA_MMV_Y)
else ifdef CHATLLM_CUDA_DMMV_Y
	MK_NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(CHATLLM_CUDA_DMMV_Y) # for backwards compatibility
else
	MK_NVCCFLAGS += -DGGML_CUDA_MMV_Y=1
endif # CHATLLM_CUDA_MMV_Y
ifdef CHATLLM_CUDA_F16
	MK_NVCCFLAGS += -DGGML_CUDA_F16
endif # CHATLLM_CUDA_F16
ifdef CHATLLM_CUDA_DMMV_F16
	MK_NVCCFLAGS += -DGGML_CUDA_F16
endif # CHATLLM_CUDA_DMMV_F16
ifdef CHATLLM_CUDA_KQUANTS_ITER
	MK_NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(CHATLLM_CUDA_KQUANTS_ITER)
else
	MK_NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif
ifdef CHATLLM_CUDA_PEER_MAX_BATCH_SIZE
	MK_NVCCFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=$(CHATLLM_CUDA_PEER_MAX_BATCH_SIZE)
else
	MK_NVCCFLAGS += -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
endif # CHATLLM_CUDA_PEER_MAX_BATCH_SIZE
ifdef CHATLLM_CUDA_NO_PEER_COPY
	MK_NVCCFLAGS += -DGGML_CUDA_NO_PEER_COPY
endif # CHATLLM_CUDA_NO_PEER_COPY
ifdef CHATLLM_CUDA_CCBIN
	MK_NVCCFLAGS += -ccbin $(CHATLLM_CUDA_CCBIN)
endif

ifdef JETSON_EOL_MODULE_DETECT
define NVCC_COMPILE
	$(NVCC) -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_CUDA -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/aarch64-linux/include -std=c++11 -O3 $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CUDA_CXXFLAGS)" -c $< -o $@
endef # NVCC_COMPILE
else
define NVCC_COMPILE
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CUDA_CXXFLAGS)" -c $< -o $@
endef # NVCC_COMPILE
endif # JETSON_EOL_MODULE_DETECT

$(OUTPUT_PATH)/ggml-cuda/%.o: ggml-cuda/%.cu ggml-cuda/%.cuh $(GGML_INC)/ggml.h $(GGML_INC)/ggml-common.h $(GGML_INC)/ggml-cuda/common.cuh
	$(NVCC_COMPILE)

$(OUTPUT_PATH)/ggml-cuda.o: $(GGML_SRC)/ggml-cuda.cu $(GGML_INC)/ggml-cuda.h $(GGML_INC)/ggml.h $(GGML_INC)/ggml-backend.h $(GGML_INC)/ggml-backend-impl.h $(GGML_INC)/ggml-common.h $(wildcard ggml-cuda/*.cuh)
	$(NVCC_COMPILE)

endif # CHATLLM_CUDA

ifdef CHATLLM_CLBLAST

	MK_CPPFLAGS += -DGGML_USE_CLBLAST $(shell pkg-config --cflags-only-I clblast OpenCL)
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other clblast OpenCL)
	MK_CXXFLAGS += $(shell pkg-config --cflags-only-other clblast OpenCL)

	# Mac provides OpenCL as a framework
	ifeq ($(UNAME_S),Darwin)
		MK_LDFLAGS += -lclblast -framework OpenCL
	else
		MK_LDFLAGS += $(shell pkg-config --libs clblast OpenCL)
	endif
	OBJS    += $(OUTPUT_PATH)/ggml-opencl.o

$(OUTPUT_PATH)/ggml-opencl.o: $(GGML_SRC)/ggml-opencl.cpp $(GGML_INC)/ggml-opencl.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif # CHATLLM_CLBLAST

ifdef CHATLLM_VULKAN
	MK_CPPFLAGS  += -DGGML_USE_VULKAN
	MK_LDFLAGS += -lvulkan
	OBJS    += $(OUTPUT_PATH)/ggml-vulkan.o

ifdef CHATLLM_VULKAN_CHECK_RESULTS
	MK_CPPFLAGS  += -DGGML_VULKAN_CHECK_RESULTS
endif

ifdef CHATLLM_VULKAN_DEBUG
	MK_CPPFLAGS  += -DGGML_VULKAN_DEBUG
endif

ifdef CHATLLM_VULKAN_VALIDATE
	MK_CPPFLAGS  += -DGGML_VULKAN_VALIDATE
endif

ifdef CHATLLM_VULKAN_RUN_TESTS
	MK_CPPFLAGS  += -DGGML_VULKAN_RUN_TESTS
endif

$(OUTPUT_PATH)/ggml-vulkan.o: $(GGML_SRC)/ggml-vulkan.cpp $(GGML_INC)/ggml-vulkan.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif # CHATLLM_VULKAN

ifdef CHATLLM_HIPBLAS
	ifeq ($(wildcard /opt/rocm),)
		ROCM_PATH	?= /usr
		GPU_TARGETS ?= $(shell $(shell which amdgpu-arch))
	else
		ROCM_PATH	?= /opt/rocm
		GPU_TARGETS ?= $(shell $(ROCM_PATH)/llvm/bin/amdgpu-arch)
	endif
	HIPCC                   ?= $(CCACHE) $(ROCM_PATH)/bin/hipcc
	CHATLLM_CUDA_DMMV_X       ?= 32
	CHATLLM_CUDA_MMV_Y        ?= 1
	CHATLLM_CUDA_KQUANTS_ITER ?= 2
	MK_CPPFLAGS += -DGGML_USE_HIPBLAS -DGGML_USE_CUDA
ifdef CHATLLM_HIP_UMA
	MK_CPPFLAGS += -DGGML_HIP_UMA
endif # CHATLLM_HIP_UMA
	MK_LDFLAGS  += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib
	MK_LDFLAGS	+= -lhipblas -lamdhip64 -lrocblas
	HIPFLAGS    += $(addprefix --offload-arch=,$(GPU_TARGETS))
	HIPFLAGS    += -DGGML_CUDA_DMMV_X=$(CHATLLM_CUDA_DMMV_X)
	HIPFLAGS    += -DGGML_CUDA_MMV_Y=$(CHATLLM_CUDA_MMV_Y)
	HIPFLAGS    += -DK_QUANTS_PER_ITERATION=$(CHATLLM_CUDA_KQUANTS_ITER)
ifdef CHATLLM_CUDA_FORCE_DMMV
	HIPFLAGS 	+= -DGGML_CUDA_FORCE_DMMV
endif # CHATLLM_CUDA_FORCE_DMMV
ifdef CHATLLM_CUDA_NO_PEER_COPY
	HIPFLAGS 	+= -DGGML_CUDA_NO_PEER_COPY
endif # CHATLLM_CUDA_NO_PEER_COPY
	OBJS        += $(OUTPUT_PATH)/ggml-cuda.o
	OBJS        += $(patsubst %.cu,%.o,$(wildcard ggml-cuda/*.cu))

$(OUTPUT_PATH)/ggml-cuda.o: $(GGML_SRC)/ggml-cuda.cu $(GGML_INC)/ggml-cuda.h $(GGML_INC)/ggml.h $(GGML_INC)/ggml-backend.h $(GGML_INC)/ggml-backend-impl.h $(GGML_INC)/ggml-common.h $(wildcard ggml-cuda/*.cuh)
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<

$(OUTPUT_PATH)/ggml-cuda/%.o: ggml-cuda/%.cu ggml-cuda/%.cuh $(GGML_SRC)/ggml.h $(GGML_SRC)/ggml-common.h ggml-cuda/common.cuh
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -x hip -c -o $@ $<

endif # CHATLLM_HIPBLAS

ifdef CHATLLM_METAL
	MK_CPPFLAGS += -DGGML_USE_METAL
	MK_LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit
	OBJS		+= $(OUTPUT_PATH)/ggml-metal.o
ifdef CHATLLM_METAL_NDEBUG
	MK_CPPFLAGS += -DGGML_METAL_NDEBUG
endif
ifdef CHATLLM_METAL_EMBED_LIBRARY
	MK_CPPFLAGS += -DGGML_METAL_EMBED_LIBRARY
	OBJS        += $(OUTPUT_PATH)/ggml-metal-embed.o
endif
endif # CHATLLM_METAL

ifdef CHATLLM_METAL
$(OUTPUT_PATH)/ggml-metal.o: $(GGML_SRC)/ggml-metal.m $(GGML_SRC)/ggml-metal.h $(GGML_SRC)/ggml.h
	$(CC) $(CFLAGS) -c $< -o $@

ifdef CHATLLM_METAL_EMBED_LIBRARY
$(OUTPUT_PATH)/ggml-metal-embed.o: ggml-metal.metal ggml-common.h
	@echo "Embedding Metal library"
	@sed -e '/#include "ggml-common.h"/r ggml-common.h' -e '/#include "ggml-common.h"/d' < ggml-metal.metal > ggml-metal-embed.metal
	$(eval TEMP_ASSEMBLY=$(shell mktemp))
	@echo ".section __DATA, __ggml_metallib"   >  $(TEMP_ASSEMBLY)
	@echo ".globl _ggml_metallib_start"        >> $(TEMP_ASSEMBLY)
	@echo "_ggml_metallib_start:"              >> $(TEMP_ASSEMBLY)
	@echo ".incbin \"ggml-metal-embed.metal\"" >> $(TEMP_ASSEMBLY)
	@echo ".globl _ggml_metallib_end"          >> $(TEMP_ASSEMBLY)
	@echo "_ggml_metallib_end:"                >> $(TEMP_ASSEMBLY)
	@$(AS) $(TEMP_ASSEMBLY) -o $@
	@rm -f ${TEMP_ASSEMBLY}
endif
endif # CHATLLM_METAL

ifdef CHATLLM_MPI
$(OUTPUT_PATH)/ggml-mpi.o: $(GGML_SRC)/ggml-mpi.c $(GGML_SRC)/ggml-mpi.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # CHATLLM_MPI

GF_CC := $(CC)
include scripts/get-flags.mk

# combine build flags with cmdline overrides
override CPPFLAGS  := $(MK_CPPFLAGS) $(CPPFLAGS)
override CFLAGS    := $(CPPFLAGS) $(MK_CFLAGS) $(GF_CFLAGS) $(CFLAGS)
BASE_CXXFLAGS      := $(MK_CXXFLAGS) $(CXXFLAGS)
override CXXFLAGS  := $(BASE_CXXFLAGS) $(HOST_CXXFLAGS) $(GF_CXXFLAGS) $(CPPFLAGS)
override NVCCFLAGS := $(MK_NVCCFLAGS) $(NVCCFLAGS)
override LDFLAGS   := $(MK_LDFLAGS) $(LDFLAGS)

# identify CUDA host compiler
ifdef CHATLLM_CUDA
GF_CC := $(NVCC) $(NVCCFLAGS) 2>/dev/null .c -Xcompiler
include scripts/get-flags.mk
CUDA_CXXFLAGS := $(BASE_CXXFLAGS) $(GF_CXXFLAGS) -Wno-pedantic
endif

ifdef CHATLLM_CURL
override CXXFLAGS := $(CXXFLAGS) -DCHATLLM_USE_CURL
override LDFLAGS  := $(LDFLAGS) -lcurl
endif

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:   $(UNAME_S))
$(info I UNAME_P:   $(UNAME_P))
$(info I UNAME_M:   $(UNAME_M))
$(info I CFLAGS:    $(CFLAGS))
$(info I CXXFLAGS:  $(CXXFLAGS))
$(info I NVCCFLAGS: $(NVCCFLAGS))
$(info I LDFLAGS:   $(LDFLAGS))
$(info I CC:        $(shell $(CC)   --version | head -n 1))
$(info I CXX:       $(shell $(CXX)  --version | head -n 1))
ifdef CHATLLM_CUDA
$(info I NVCC:      $(shell $(NVCC) --version | tail -n 1))
CUDA_VERSION := $(shell $(NVCC) --version | grep -oP 'release (\K[0-9]+\.[0-9])')
ifeq ($(shell awk -v "v=$(CUDA_VERSION)" 'BEGIN { print (v < 11.7) }'),1)
ifndef CUDA_DOCKER_ARCH
ifndef CUDA_POWER_ARCH
$(error I ERROR: For CUDA versions < 11.7 a target CUDA architecture must be explicitly provided via environment variable CUDA_DOCKER_ARCH, e.g. by running "export CUDA_DOCKER_ARCH=compute_XX" on Unix-like systems, where XX is the minimum compute capability that the code needs to run on. A list with compute capabilities can be found here: https://developer.nvidia.com/cuda-gpus )
endif # CUDA_POWER_ARCH
endif # CUDA_DOCKER_ARCH
endif # eq ($(shell echo "$(CUDA_VERSION) < 11.7" | bc),1)
endif # CHATLLM_CUDA
$(info )

ifdef CHATLLM_CUBLAS
$(info !!!!)
$(info CHATLLM_CUBLAS is deprecated and will be removed in the future. Use CHATLLM_CUDA instead.)
$(info !!!!)
$(info )
endif

#
# Build library
#

$(OUTPUT_PATH)/ggml.o: $(GGML_SRC)/ggml.c $(GGML_INC)/ggml.h $(GGML_SRC)/ggml-cuda.h
	$(CC)  $(CFLAGS)   -c $< -o $@

$(OUTPUT_PATH)/ggml-alloc.o: $(GGML_SRC)/ggml-alloc.c $(GGML_INC)/ggml.h $(GGML_INC)/ggml-alloc.h
	$(CC)  $(CFLAGS)   -c $< -o $@

$(OUTPUT_PATH)/ggml-backend.o: $(GGML_SRC)/ggml-backend.c $(GGML_INC)/ggml.h $(GGML_INC)/ggml-backend.h
	$(CC)  $(CFLAGS)   -c $< -o $@

$(OUTPUT_PATH)/ggml-quants.o: $(GGML_SRC)/ggml-quants.c $(GGML_INC)/ggml.h $(GGML_SRC)/ggml-quants.h $(GGML_SRC)/ggml-common.h
	$(CC) $(CFLAGS)    -c $< -o $@

$(OUTPUT_PATH)/tokenizer.o: tokenizer.cpp unicode.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUTPUT_PATH)/vectorstore.o: vectorstore.cpp vectorstore.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUTPUT_PATH)/chat.o: chat.cpp chat.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUTPUT_PATH)/models.o: models.cpp models.h unicode.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUTPUT_PATH)/layers.o:layers.cpp layers.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

OBJS += $(OUTPUT_PATH)/ggml-alloc.o $(OUTPUT_PATH)/ggml-backend.o $(OUTPUT_PATH)/ggml-quants.o

COMMON_H_DEPS =
COMMON_DEPS   =

$(OUTPUT_PATH)/libchatllm.so: $(OUTPUT_PATH)/models.o $(OUTPUT_PATH)/chat.o $(OUTPUT_PATH)/vectorstore.o $(OUTPUT_PATH)/tokenizer.o $(OUTPUT_PATH)/ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

$(OUTPUT_PATH)/libchatllm.a: $(OUTPUT_PATH)/models.o $(OUTPUT_PATH)/chat.o $(OUTPUT_PATH)/vectorstore.o $(OUTPUT_PATH)/tokenizer.o $(OUTPUT_PATH)/ggml.o $(OBJS) $(COMMON_DEPS)
	ar rcs libchatllm.a $(OUTPUT_PATH)/models.o $(OUTPUT_PATH)/chat.o $(OUTPUT_PATH)/vectorstore.o $(OUTPUT_PATH)/tokenizer.o $(OUTPUT_PATH)/ggml.o $(OBJS) $(COMMON_DEPS)

clean:
	rm -vrf $(OUTPUT_PATH)/*.o  $(OBJ_PATH)/*.so $(OBJ_PATH)/*.a $(OBJ_PATH)/*.dll *.dot $(BUILD_TARGETS)
	rm -vrf ggml-cuda/*.o

#
# Examples
#

# $< is the first prerequisite, i.e. the source file.
# Explicitly compile this to an object file so that it can be cached with ccache.
# The source file is then filtered out from $^ (the list of all prerequisites) and the object file is added instead.

# Helper function that replaces .c, .cpp, and .cu file endings with .o:
GET_OBJ_FILE = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(1))))

$(OUTPUT_PATH)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


$(OUTPUT_PATH)/main: $(OUTPUT_PATH)/main.o $(OUTPUT_PATH)/models.o $(OUTPUT_PATH)/chat.o $(OUTPUT_PATH)/layers.o $(OUTPUT_PATH)/vectorstore.o $(OUTPUT_PATH)/tokenizer.o $(OUTPUT_PATH)/ggml.o $(COMMON_DEPS) $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./obj/main -h for help.  ===='
	@echo

ifeq ($(UNAME_S),Darwin)
swift: examples/batched.swift
	(cd examples/batched.swift; make build)
endif
