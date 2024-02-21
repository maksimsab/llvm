// REQUIRES: system-linux
// This test check wrapping of SYCL binaries in clang-linker-wrapper.

// RUN: %clang -cc1 -fsycl-is-device -disable-llvm-passes -triple=spir64-unknown-unknown %s -emit-llvm-bc -o %t.device.bc
// RUN: clang-offload-packager -o %t.fat --image=file=%t.device.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple=x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.fat
// RUN: clang-linker-wrapper --print-wrapped-module --host-triple=x86_64-unknown-linux-gnu --triple=spir64 \
// RUN:                      -sycl-device-library-location=%S/Inputs \
// RUN:                      -sycl-post-link-split-mode=auto \
// RUN:                      %t.o -o %t.out 2>&1 --linker-path="/usr/bin/ld" | FileCheck %s

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
    func();
}

extern "C" {
// symbols so that linker find them and doesn't fail.
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

int main() {
    kernel<class fake_kernel>([](){});
}

//#endif
