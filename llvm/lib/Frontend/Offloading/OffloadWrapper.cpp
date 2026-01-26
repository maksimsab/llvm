//===- OffloadWrapper.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Frontend/Offloading/PropertySet.h"
//#include "llvm/Support/PropertySetIO.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"

#include <variant>
#include <memory>
#include <utility>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::offloading;

namespace {
/// Magic number that begins the section containing the CUDA fatbinary.
constexpr unsigned CudaFatMagic = 0x466243b1;
constexpr unsigned HIPFatMagic = 0x48495046;

IntegerType *getSizeTTy(Module &M) {
  return M.getDataLayout().getIntPtrType(M.getContext());
}

// struct __tgt_device_image {
//   void *ImageStart;
//   void *ImageEnd;
//   __tgt_offload_entry *EntriesBegin;
//   __tgt_offload_entry *EntriesEnd;
// };
StructType *getDeviceImageTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *ImageTy = StructType::getTypeByName(C, "__tgt_device_image");
  if (!ImageTy)
    ImageTy =
        StructType::create("__tgt_device_image", PointerType::getUnqual(C),
                           PointerType::getUnqual(C), PointerType::getUnqual(C),
                           PointerType::getUnqual(C));
  return ImageTy;
}

PointerType *getDeviceImagePtrTy(Module &M) {
  return PointerType::getUnqual(M.getContext());
}

// struct __tgt_bin_desc {
//   int32_t NumDeviceImages;
//   __tgt_device_image *DeviceImages;
//   __tgt_offload_entry *HostEntriesBegin;
//   __tgt_offload_entry *HostEntriesEnd;
// };
StructType *getBinDescTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *DescTy = StructType::getTypeByName(C, "__tgt_bin_desc");
  if (!DescTy)
    DescTy = StructType::create(
        "__tgt_bin_desc", Type::getInt32Ty(C), getDeviceImagePtrTy(M),
        PointerType::getUnqual(C), PointerType::getUnqual(C));
  return DescTy;
}

PointerType *getBinDescPtrTy(Module &M) {
  return PointerType::getUnqual(M.getContext());
}

/// Creates binary descriptor for the given device images. Binary descriptor
/// is an object that is passed to the offloading runtime at program startup
/// and it describes all device images available in the executable or shared
/// library. It is defined as follows
///
/// __attribute__((visibility("hidden")))
/// extern __tgt_offload_entry *__start_omp_offloading_entries;
/// __attribute__((visibility("hidden")))
/// extern __tgt_offload_entry *__stop_omp_offloading_entries;
///
/// static const char Image0[] = { <Bufs.front() contents> };
///  ...
/// static const char ImageN[] = { <Bufs.back() contents> };
///
/// static const __tgt_device_image Images[] = {
///   {
///     Image0,                            /*ImageStart*/
///     Image0 + sizeof(Image0),           /*ImageEnd*/
///     __start_omp_offloading_entries,    /*EntriesBegin*/
///     __stop_omp_offloading_entries      /*EntriesEnd*/
///   },
///   ...
///   {
///     ImageN,                            /*ImageStart*/
///     ImageN + sizeof(ImageN),           /*ImageEnd*/
///     __start_omp_offloading_entries,    /*EntriesBegin*/
///     __stop_omp_offloading_entries      /*EntriesEnd*/
///   }
/// };
///
/// static const __tgt_bin_desc BinDesc = {
///   sizeof(Images) / sizeof(Images[0]),  /*NumDeviceImages*/
///   Images,                              /*DeviceImages*/
///   __start_omp_offloading_entries,      /*HostEntriesBegin*/
///   __stop_omp_offloading_entries        /*HostEntriesEnd*/
/// };
///
/// Global variable that represents BinDesc is returned.
GlobalVariable *createBinDesc(Module &M, ArrayRef<ArrayRef<char>> Bufs,
                              EntryArrayTy EntryArray, StringRef Suffix,
                              bool Relocatable) {
  LLVMContext &C = M.getContext();
  auto [EntriesB, EntriesE] = EntryArray;

  auto *Zero = ConstantInt::get(getSizeTTy(M), 0u);
  Constant *ZeroZero[] = {Zero, Zero};

  // Create initializer for the images array.
  SmallVector<Constant *, 4u> ImagesInits;
  ImagesInits.reserve(Bufs.size());
  for (ArrayRef<char> Buf : Bufs) {
    // We embed the full offloading entry so the binary utilities can parse it.
    auto *Data = ConstantDataArray::get(C, Buf);
    auto *Image = new GlobalVariable(M, Data->getType(), /*isConstant=*/true,
                                     GlobalVariable::InternalLinkage, Data,
                                     ".omp_offloading.device_image" + Suffix);
    Image->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    Image->setSection(Relocatable ? ".llvm.offloading.relocatable"
                                  : ".llvm.offloading");
    Image->setAlignment(Align(object::OffloadBinary::getAlignment()));

    StringRef Binary(Buf.data(), Buf.size());

    uint64_t BeginOffset = 0;
    uint64_t EndOffset = Binary.size();

    // Optionally use an offload binary for its offload dumping support.
    // The device image struct contains the pointer to the beginning and end of
    // the image stored inside of the offload binary. There should only be one
    // of these for each buffer so we parse it out manually.
    if (identify_magic(Binary) == file_magic::offload_binary) {
      const auto *Header =
          reinterpret_cast<const object::OffloadBinary::Header *>(
              Binary.bytes_begin());
      const auto *Entry =
          reinterpret_cast<const object::OffloadBinary::Entry *>(
              Binary.bytes_begin() + Header->EntryOffset);
      BeginOffset = Entry->ImageOffset;
      EndOffset = Entry->ImageOffset + Entry->ImageSize;
    }

    auto *Begin = ConstantInt::get(getSizeTTy(M), BeginOffset);
    auto *Size = ConstantInt::get(getSizeTTy(M), EndOffset);
    Constant *ZeroBegin[] = {Zero, Begin};
    Constant *ZeroSize[] = {Zero, Size};

    auto *ImageB =
        ConstantExpr::getGetElementPtr(Image->getValueType(), Image, ZeroBegin);
    auto *ImageE =
        ConstantExpr::getGetElementPtr(Image->getValueType(), Image, ZeroSize);

    ImagesInits.push_back(ConstantStruct::get(getDeviceImageTy(M), ImageB,
                                              ImageE, EntriesB, EntriesE));
  }

  // Then create images array.
  auto *ImagesData = ConstantArray::get(
      ArrayType::get(getDeviceImageTy(M), ImagesInits.size()), ImagesInits);

  auto *Images =
      new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                         GlobalValue::InternalLinkage, ImagesData,
                         ".omp_offloading.device_images" + Suffix);
  Images->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  auto *ImagesB =
      ConstantExpr::getGetElementPtr(Images->getValueType(), Images, ZeroZero);

  // And finally create the binary descriptor object.
  auto *DescInit = ConstantStruct::get(
      getBinDescTy(M),
      ConstantInt::get(Type::getInt32Ty(C), ImagesInits.size()), ImagesB,
      EntriesB, EntriesE);

  return new GlobalVariable(M, DescInit->getType(), /*isConstant=*/true,
                            GlobalValue::InternalLinkage, DescInit,
                            ".omp_offloading.descriptor" + Suffix);
}

Function *createUnregisterFunction(Module &M, GlobalVariable *BinDesc,
                                   StringRef Suffix) {
  LLVMContext &C = M.getContext();
  auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *Func =
      Function::Create(FuncTy, GlobalValue::InternalLinkage,
                       ".omp_offloading.descriptor_unreg" + Suffix, &M);
  Func->setSection(".text.startup");

  // Get __tgt_unregister_lib function declaration.
  auto *UnRegFuncTy = FunctionType::get(Type::getVoidTy(C), getBinDescPtrTy(M),
                                        /*isVarArg*/ false);
  FunctionCallee UnRegFuncC =
      M.getOrInsertFunction("__tgt_unregister_lib", UnRegFuncTy);

  // Construct function body
  IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
  Builder.CreateCall(UnRegFuncC, BinDesc);
  Builder.CreateRetVoid();

  return Func;
}

void createRegisterFunction(Module &M, GlobalVariable *BinDesc,
                            StringRef Suffix) {
  LLVMContext &C = M.getContext();
  auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                ".omp_offloading.descriptor_reg" + Suffix, &M);
  Func->setSection(".text.startup");

  // Get __tgt_register_lib function declaration.
  auto *RegFuncTy = FunctionType::get(Type::getVoidTy(C), getBinDescPtrTy(M),
                                      /*isVarArg*/ false);
  FunctionCallee RegFuncC =
      M.getOrInsertFunction("__tgt_register_lib", RegFuncTy);

  auto *AtExitTy = FunctionType::get(
      Type::getInt32Ty(C), PointerType::getUnqual(C), /*isVarArg=*/false);
  FunctionCallee AtExit = M.getOrInsertFunction("atexit", AtExitTy);

  Function *UnregFunc = createUnregisterFunction(M, BinDesc, Suffix);

  // Construct function body
  IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));

  Builder.CreateCall(RegFuncC, BinDesc);

  // Register the destructors with 'atexit'. This is expected by the CUDA
  // runtime and ensures that we clean up before dynamic objects are destroyed.
  // This needs to be done after plugin initialization to ensure that it is
  // called before the plugin runtime is destroyed.
  Builder.CreateCall(AtExit, UnregFunc);
  Builder.CreateRetVoid();

  // Add this function to constructors.
  appendToGlobalCtors(M, Func, /*Priority=*/101);
}

// struct fatbin_wrapper {
//  int32_t magic;
//  int32_t version;
//  void *image;
//  void *reserved;
//};
StructType *getFatbinWrapperTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *FatbinTy = StructType::getTypeByName(C, "fatbin_wrapper");
  if (!FatbinTy)
    FatbinTy = StructType::create(
        "fatbin_wrapper", Type::getInt32Ty(C), Type::getInt32Ty(C),
        PointerType::getUnqual(C), PointerType::getUnqual(C));
  return FatbinTy;
}

/// Embed the image \p Image into the module \p M so it can be found by the
/// runtime.
GlobalVariable *createFatbinDesc(Module &M, ArrayRef<char> Image, bool IsHIP,
                                 StringRef Suffix) {
  LLVMContext &C = M.getContext();
  llvm::Type *Int8PtrTy = PointerType::getUnqual(C);
  const llvm::Triple &Triple = M.getTargetTriple();

  // Create the global string containing the fatbinary.
  StringRef FatbinConstantSection =
      IsHIP ? ".hip_fatbin"
            : (Triple.isMacOSX() ? "__NV_CUDA,__nv_fatbin" : ".nv_fatbin");
  auto *Data = ConstantDataArray::get(C, Image);
  auto *Fatbin = new GlobalVariable(M, Data->getType(), /*isConstant*/ true,
                                    GlobalVariable::InternalLinkage, Data,
                                    ".fatbin_image" + Suffix);
  Fatbin->setSection(FatbinConstantSection);

  // Create the fatbinary wrapper
  StringRef FatbinWrapperSection = IsHIP               ? ".hipFatBinSegment"
                                   : Triple.isMacOSX() ? "__NV_CUDA,__fatbin"
                                                       : ".nvFatBinSegment";
  Constant *FatbinWrapper[] = {
      ConstantInt::get(Type::getInt32Ty(C), IsHIP ? HIPFatMagic : CudaFatMagic),
      ConstantInt::get(Type::getInt32Ty(C), 1),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Fatbin, Int8PtrTy),
      ConstantPointerNull::get(PointerType::getUnqual(C))};

  Constant *FatbinInitializer =
      ConstantStruct::get(getFatbinWrapperTy(M), FatbinWrapper);

  auto *FatbinDesc =
      new GlobalVariable(M, getFatbinWrapperTy(M),
                         /*isConstant*/ true, GlobalValue::InternalLinkage,
                         FatbinInitializer, ".fatbin_wrapper" + Suffix);
  FatbinDesc->setSection(FatbinWrapperSection);
  FatbinDesc->setAlignment(Align(8));

  return FatbinDesc;
}

/// Create the register globals function. We will iterate all of the offloading
/// entries stored at the begin / end symbols and register them according to
/// their type. This creates the following function in IR:
///
/// extern struct __tgt_offload_entry __start_cuda_offloading_entries;
/// extern struct __tgt_offload_entry __stop_cuda_offloading_entries;
///
/// extern void __cudaRegisterFunction(void **, void *, void *, void *, int,
///                                    void *, void *, void *, void *, int *);
/// extern void __cudaRegisterVar(void **, void *, void *, void *, int32_t,
///                               int64_t, int32_t, int32_t);
///
/// void __cudaRegisterTest(void **fatbinHandle) {
///   for (struct __tgt_offload_entry *entry = &__start_cuda_offloading_entries;
///        entry != &__stop_cuda_offloading_entries; ++entry) {
///     if (entry->Kind != OFK_CUDA)
///       continue
///
///     if (!entry->Size)
///       __cudaRegisterFunction(fatbinHandle, entry->addr, entry->name,
///                              entry->name, -1, 0, 0, 0, 0, 0);
///     else
///       __cudaRegisterVar(fatbinHandle, entry->addr, entry->name, entry->name,
///                         0, entry->size, 0, 0);
///   }
/// }
Function *createRegisterGlobalsFunction(Module &M, bool IsHIP,
                                        EntryArrayTy EntryArray,
                                        StringRef Suffix,
                                        bool EmitSurfacesAndTextures) {
  LLVMContext &C = M.getContext();
  auto [EntriesB, EntriesE] = EntryArray;

  // Get the __cudaRegisterFunction function declaration.
  PointerType *Int8PtrTy = PointerType::get(C, 0);
  PointerType *Int8PtrPtrTy = PointerType::get(C, 0);
  PointerType *Int32PtrTy = PointerType::get(C, 0);
  auto *RegFuncTy = FunctionType::get(
      Type::getInt32Ty(C),
      {Int8PtrPtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy, Type::getInt32Ty(C),
       Int8PtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy, Int32PtrTy},
      /*isVarArg*/ false);
  FunctionCallee RegFunc = M.getOrInsertFunction(
      IsHIP ? "__hipRegisterFunction" : "__cudaRegisterFunction", RegFuncTy);

  // Get the __cudaRegisterVar function declaration.
  auto *RegVarTy = FunctionType::get(
      Type::getVoidTy(C),
      {Int8PtrPtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy, Type::getInt32Ty(C),
       getSizeTTy(M), Type::getInt32Ty(C), Type::getInt32Ty(C)},
      /*isVarArg*/ false);
  FunctionCallee RegVar = M.getOrInsertFunction(
      IsHIP ? "__hipRegisterVar" : "__cudaRegisterVar", RegVarTy);

  // Get the __cudaRegisterSurface function declaration.
  FunctionType *RegManagedVarTy =
      FunctionType::get(Type::getVoidTy(C),
                        {Int8PtrPtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy,
                         getSizeTTy(M), Type::getInt32Ty(C)},
                        /*isVarArg=*/false);
  FunctionCallee RegManagedVar = M.getOrInsertFunction(
      IsHIP ? "__hipRegisterManagedVar" : "__cudaRegisterManagedVar",
      RegManagedVarTy);

  // Get the __cudaRegisterSurface function declaration.
  FunctionType *RegSurfaceTy =
      FunctionType::get(Type::getVoidTy(C),
                        {Int8PtrPtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy,
                         Type::getInt32Ty(C), Type::getInt32Ty(C)},
                        /*isVarArg=*/false);
  FunctionCallee RegSurface = M.getOrInsertFunction(
      IsHIP ? "__hipRegisterSurface" : "__cudaRegisterSurface", RegSurfaceTy);

  // Get the __cudaRegisterTexture function declaration.
  FunctionType *RegTextureTy = FunctionType::get(
      Type::getVoidTy(C),
      {Int8PtrPtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy, Type::getInt32Ty(C),
       Type::getInt32Ty(C), Type::getInt32Ty(C)},
      /*isVarArg=*/false);
  FunctionCallee RegTexture = M.getOrInsertFunction(
      IsHIP ? "__hipRegisterTexture" : "__cudaRegisterTexture", RegTextureTy);

  auto *RegGlobalsTy = FunctionType::get(Type::getVoidTy(C), Int8PtrPtrTy,
                                         /*isVarArg*/ false);
  auto *RegGlobalsFn =
      Function::Create(RegGlobalsTy, GlobalValue::InternalLinkage,
                       IsHIP ? ".hip.globals_reg" : ".cuda.globals_reg", &M);
  RegGlobalsFn->setSection(".text.startup");

  // Create the loop to register all the entries.
  IRBuilder<> Builder(BasicBlock::Create(C, "entry", RegGlobalsFn));
  auto *EntryBB = BasicBlock::Create(C, "while.entry", RegGlobalsFn);
  auto *IfKindBB = BasicBlock::Create(C, "if.kind", RegGlobalsFn);
  auto *IfThenBB = BasicBlock::Create(C, "if.then", RegGlobalsFn);
  auto *IfElseBB = BasicBlock::Create(C, "if.else", RegGlobalsFn);
  auto *SwGlobalBB = BasicBlock::Create(C, "sw.global", RegGlobalsFn);
  auto *SwManagedBB = BasicBlock::Create(C, "sw.managed", RegGlobalsFn);
  auto *SwSurfaceBB = BasicBlock::Create(C, "sw.surface", RegGlobalsFn);
  auto *SwTextureBB = BasicBlock::Create(C, "sw.texture", RegGlobalsFn);
  auto *IfEndBB = BasicBlock::Create(C, "if.end", RegGlobalsFn);
  auto *ExitBB = BasicBlock::Create(C, "while.end", RegGlobalsFn);

  auto *EntryCmp = Builder.CreateICmpNE(EntriesB, EntriesE);
  Builder.CreateCondBr(EntryCmp, EntryBB, ExitBB);
  Builder.SetInsertPoint(EntryBB);
  auto *Entry = Builder.CreatePHI(PointerType::getUnqual(C), 2, "entry");
  auto *AddrPtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 4)});
  auto *Addr = Builder.CreateLoad(Int8PtrTy, AddrPtr, "addr");
  auto *AuxAddrPtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 8)});
  auto *AuxAddr = Builder.CreateLoad(Int8PtrTy, AuxAddrPtr, "aux_addr");
  auto *KindPtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 2)});
  auto *Kind = Builder.CreateLoad(Type::getInt16Ty(C), KindPtr, "kind");
  auto *NamePtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 5)});
  auto *Name = Builder.CreateLoad(Int8PtrTy, NamePtr, "name");
  auto *SizePtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 6)});
  auto *Size = Builder.CreateLoad(Type::getInt64Ty(C), SizePtr, "size");
  auto *FlagsPtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 3)});
  auto *Flags = Builder.CreateLoad(Type::getInt32Ty(C), FlagsPtr, "flags");
  auto *DataPtr =
      Builder.CreateInBoundsGEP(offloading::getEntryTy(M), Entry,
                                {ConstantInt::get(Type::getInt32Ty(C), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 7)});
  auto *Data = Builder.CreateTrunc(
      Builder.CreateLoad(Type::getInt64Ty(C), DataPtr, "data"),
      Type::getInt32Ty(C));
  auto *Type = Builder.CreateAnd(
      Flags, ConstantInt::get(Type::getInt32Ty(C), 0x7), "type");

  // Extract the flags stored in the bit-field and convert them to C booleans.
  auto *ExternBit = Builder.CreateAnd(
      Flags, ConstantInt::get(Type::getInt32Ty(C),
                              llvm::offloading::OffloadGlobalExtern));
  auto *Extern = Builder.CreateLShr(
      ExternBit, ConstantInt::get(Type::getInt32Ty(C), 3), "extern");
  auto *ConstantBit = Builder.CreateAnd(
      Flags, ConstantInt::get(Type::getInt32Ty(C),
                              llvm::offloading::OffloadGlobalConstant));
  auto *Const = Builder.CreateLShr(
      ConstantBit, ConstantInt::get(Type::getInt32Ty(C), 4), "constant");
  auto *NormalizedBit = Builder.CreateAnd(
      Flags, ConstantInt::get(Type::getInt32Ty(C),
                              llvm::offloading::OffloadGlobalNormalized));
  auto *Normalized = Builder.CreateLShr(
      NormalizedBit, ConstantInt::get(Type::getInt32Ty(C), 5), "normalized");
  auto *KindCond = Builder.CreateICmpEQ(
      Kind, ConstantInt::get(Type::getInt16Ty(C),
                             IsHIP ? object::OffloadKind::OFK_HIP
                                   : object::OffloadKind::OFK_Cuda));
  Builder.CreateCondBr(KindCond, IfKindBB, IfEndBB);
  Builder.SetInsertPoint(IfKindBB);
  auto *FnCond = Builder.CreateICmpEQ(
      Size, ConstantInt::getNullValue(Type::getInt64Ty(C)));
  Builder.CreateCondBr(FnCond, IfThenBB, IfElseBB);

  // Create kernel registration code.
  Builder.SetInsertPoint(IfThenBB);
  Builder.CreateCall(RegFunc, {RegGlobalsFn->arg_begin(), Addr, Name, Name,
                               ConstantInt::get(Type::getInt32Ty(C), -1),
                               ConstantPointerNull::get(Int8PtrTy),
                               ConstantPointerNull::get(Int8PtrTy),
                               ConstantPointerNull::get(Int8PtrTy),
                               ConstantPointerNull::get(Int8PtrTy),
                               ConstantPointerNull::get(Int32PtrTy)});
  Builder.CreateBr(IfEndBB);
  Builder.SetInsertPoint(IfElseBB);

  auto *Switch = Builder.CreateSwitch(Type, IfEndBB);
  // Create global variable registration code.
  Builder.SetInsertPoint(SwGlobalBB);
  Builder.CreateCall(RegVar,
                     {RegGlobalsFn->arg_begin(), Addr, Name, Name, Extern, Size,
                      Const, ConstantInt::get(Type::getInt32Ty(C), 0)});
  Builder.CreateBr(IfEndBB);
  Switch->addCase(Builder.getInt32(llvm::offloading::OffloadGlobalEntry),
                  SwGlobalBB);

  // Create managed variable registration code.
  Builder.SetInsertPoint(SwManagedBB);
  Builder.CreateCall(RegManagedVar, {RegGlobalsFn->arg_begin(), AuxAddr, Addr,
                                     Name, Size, Data});
  Builder.CreateBr(IfEndBB);
  Switch->addCase(Builder.getInt32(llvm::offloading::OffloadGlobalManagedEntry),
                  SwManagedBB);
  // Create surface variable registration code.
  Builder.SetInsertPoint(SwSurfaceBB);
  if (EmitSurfacesAndTextures)
    Builder.CreateCall(RegSurface, {RegGlobalsFn->arg_begin(), Addr, Name, Name,
                                    Data, Extern});
  Builder.CreateBr(IfEndBB);
  Switch->addCase(Builder.getInt32(llvm::offloading::OffloadGlobalSurfaceEntry),
                  SwSurfaceBB);

  // Create texture variable registration code.
  Builder.SetInsertPoint(SwTextureBB);
  if (EmitSurfacesAndTextures)
    Builder.CreateCall(RegTexture, {RegGlobalsFn->arg_begin(), Addr, Name, Name,
                                    Data, Normalized, Extern});
  Builder.CreateBr(IfEndBB);
  Switch->addCase(Builder.getInt32(llvm::offloading::OffloadGlobalTextureEntry),
                  SwTextureBB);

  Builder.SetInsertPoint(IfEndBB);
  auto *NewEntry = Builder.CreateInBoundsGEP(
      offloading::getEntryTy(M), Entry, ConstantInt::get(getSizeTTy(M), 1));
  auto *Cmp = Builder.CreateICmpEQ(
      NewEntry,
      ConstantExpr::getInBoundsGetElementPtr(
          ArrayType::get(offloading::getEntryTy(M), 0), EntriesE,
          ArrayRef<Constant *>({ConstantInt::get(getSizeTTy(M), 0),
                                ConstantInt::get(getSizeTTy(M), 0)})));
  Entry->addIncoming(
      ConstantExpr::getInBoundsGetElementPtr(
          ArrayType::get(offloading::getEntryTy(M), 0), EntriesB,
          ArrayRef<Constant *>({ConstantInt::get(getSizeTTy(M), 0),
                                ConstantInt::get(getSizeTTy(M), 0)})),
      &RegGlobalsFn->getEntryBlock());
  Entry->addIncoming(NewEntry, IfEndBB);
  Builder.CreateCondBr(Cmp, ExitBB, EntryBB);
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  return RegGlobalsFn;
}

// Create the constructor and destructor to register the fatbinary with the CUDA
// runtime.
void createRegisterFatbinFunction(Module &M, GlobalVariable *FatbinDesc,
                                  bool IsHIP, EntryArrayTy EntryArray,
                                  StringRef Suffix,
                                  bool EmitSurfacesAndTextures) {
  LLVMContext &C = M.getContext();
  auto *CtorFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *CtorFunc = Function::Create(
      CtorFuncTy, GlobalValue::InternalLinkage,
      (IsHIP ? ".hip.fatbin_reg" : ".cuda.fatbin_reg") + Suffix, &M);
  CtorFunc->setSection(".text.startup");

  auto *DtorFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *DtorFunc = Function::Create(
      DtorFuncTy, GlobalValue::InternalLinkage,
      (IsHIP ? ".hip.fatbin_unreg" : ".cuda.fatbin_unreg") + Suffix, &M);
  DtorFunc->setSection(".text.startup");

  auto *PtrTy = PointerType::getUnqual(C);

  // Get the __cudaRegisterFatBinary function declaration.
  auto *RegFatTy = FunctionType::get(PtrTy, PtrTy, /*isVarArg=*/false);
  FunctionCallee RegFatbin = M.getOrInsertFunction(
      IsHIP ? "__hipRegisterFatBinary" : "__cudaRegisterFatBinary", RegFatTy);
  // Get the __cudaRegisterFatBinaryEnd function declaration.
  auto *RegFatEndTy =
      FunctionType::get(Type::getVoidTy(C), PtrTy, /*isVarArg=*/false);
  FunctionCallee RegFatbinEnd =
      M.getOrInsertFunction("__cudaRegisterFatBinaryEnd", RegFatEndTy);
  // Get the __cudaUnregisterFatBinary function declaration.
  auto *UnregFatTy =
      FunctionType::get(Type::getVoidTy(C), PtrTy, /*isVarArg=*/false);
  FunctionCallee UnregFatbin = M.getOrInsertFunction(
      IsHIP ? "__hipUnregisterFatBinary" : "__cudaUnregisterFatBinary",
      UnregFatTy);

  auto *AtExitTy =
      FunctionType::get(Type::getInt32Ty(C), PtrTy, /*isVarArg=*/false);
  FunctionCallee AtExit = M.getOrInsertFunction("atexit", AtExitTy);

  auto *BinaryHandleGlobal = new llvm::GlobalVariable(
      M, PtrTy, false, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantPointerNull::get(PtrTy),
      (IsHIP ? ".hip.binary_handle" : ".cuda.binary_handle") + Suffix);

  // Create the constructor to register this image with the runtime.
  IRBuilder<> CtorBuilder(BasicBlock::Create(C, "entry", CtorFunc));
  CallInst *Handle = CtorBuilder.CreateCall(
      RegFatbin,
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(FatbinDesc, PtrTy));
  CtorBuilder.CreateAlignedStore(
      Handle, BinaryHandleGlobal,
      Align(M.getDataLayout().getPointerTypeSize(PtrTy)));
  CtorBuilder.CreateCall(createRegisterGlobalsFunction(M, IsHIP, EntryArray,
                                                       Suffix,
                                                       EmitSurfacesAndTextures),
                         Handle);
  if (!IsHIP)
    CtorBuilder.CreateCall(RegFatbinEnd, Handle);
  CtorBuilder.CreateCall(AtExit, DtorFunc);
  CtorBuilder.CreateRetVoid();

  // Create the destructor to unregister the image with the runtime. We cannot
  // use a standard global destructor after CUDA 9.2 so this must be called by
  // `atexit()` instead.
  IRBuilder<> DtorBuilder(BasicBlock::Create(C, "entry", DtorFunc));
  LoadInst *BinaryHandle = DtorBuilder.CreateAlignedLoad(
      PtrTy, BinaryHandleGlobal,
      Align(M.getDataLayout().getPointerTypeSize(PtrTy)));
  DtorBuilder.CreateCall(UnregFatbin, BinaryHandle);
  DtorBuilder.CreateRetVoid();

  // Add this function to constructors.
  appendToGlobalCtors(M, CtorFunc, /*Priority=*/101);
}

/// SYCLWrapper helper class that creates all LLVM IRs wrapping given images.
class SYCLWrapper {
public:
  SYCLWrapper(Module &M, const SYCLJITOptions &Options)
      : M(M), C(M.getContext()), Options(Options) {
    SyclPropTy = getSyclPropTy();
    SyclPropSetTy = getSyclPropSetTy();
    EntryTy = offloading::getEntryTy(M);
    SyclDeviceImageTy = getSyclDeviceImageTy();
    SyclBinDescTy = getSyclBinDescTy();
  }

  /// Creates binary descriptor for the given device images. Binary descriptor
  /// is an object that is passed to the offloading runtime at program startup
  /// and it describes all device images available in the executable or shared
  /// library. It is defined as follows:
  ///
  /// \code
  /// __attribute__((visibility("hidden")))
  /// __tgt_offload_entry *__sycl_offload_entries_arr0[];
  /// ...
  /// __attribute__((visibility("hidden")))
  /// __tgt_offload_entry *__sycl_offload_entries_arrN[];
  ///
  /// __attribute__((visibility("hidden")))
  /// extern const char *CompileOptions = "...";
  /// ...
  /// __attribute__((visibility("hidden")))
  /// extern const char *LinkOptions = "...";
  /// ...
  ///
  /// static const char Image0[] = { ... };
  ///  ...
  /// static const char ImageN[] = { ... };
  ///
  /// static const __sycl.tgt_device_image Images[] = {
  ///   {
  ///     Version,                                      // Version
  ///     OffloadKind,                                  // OffloadKind
  ///     Format,                                       // Format of the image.
  //      TripleString,                                 // Arch
  ///     CompileOptions,                               // CompileOptions
  ///     LinkOptions,                                  // LinkOptions
  ///     Image0,                                       // ImageStart
  ///     Image0 + IMAGE0_SIZE,                         // ImageEnd
  ///     __sycl_offload_entries_arr0,                  // EntriesBegin
  ///     __sycl_offload_entries_arr0 + ENTRIES0_SIZE,  // EntriesEnd
  ///     NULL,                                         // PropertiesBegin
  ///     NULL,                                         // PropertiesEnd
  ///   },
  ///   ...
  /// };
  ///
  /// static const __sycl.tgt_bin_desc FatbinDesc = {
  ///   Version,                             //Version
  ///   sizeof(Images) / sizeof(Images[0]),  //NumDeviceImages
  ///   Images,                              //DeviceImages
  ///   NULL,                                //HostEntriesBegin
  ///   NULL                                 //HostEntriesEnd
  /// };
  /// \endcode
  ///
  /// \returns Global variable that represents FatbinDesc.
  Expected<GlobalVariable *> createFatbinDesc(ArrayRef<OffloadFile> OffloadFiles) {
    StringRef OffloadKindTag = ".sycl_offloading.";
    SmallVector<Constant *> WrappedImages;
    WrappedImages.reserve(OffloadFiles.size());
    for (size_t I = 0, E = OffloadFiles.size(); I != E; ++I) {
      Expected<Constant *> ImageOrErr = wrapImage(*OffloadFiles[I].getBinary(), Twine(I), OffloadKindTag);
      if (!ImageOrErr)
        return ImageOrErr.takeError();
  
      WrappedImages.push_back(*ImageOrErr);
    }

    return combineWrappedImages(WrappedImages, OffloadKindTag);
  }

  void createRegisterFatbinFunction(GlobalVariable *FatbinDesc) {
    FunctionType *FuncTy =
        FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    Function *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                      Twine("sycl") + ".descriptor_reg", &M);
    Func->setSection(".text.startup");

    // Get RegFuncName function declaration.
    FunctionType *RegFuncTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C),
                          /*isVarArg=*/false);
    FunctionCallee RegFuncC =
        M.getOrInsertFunction("__sycl_register_lib", RegFuncTy);

    // Construct function body.
    IRBuilder Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(RegFuncC, FatbinDesc);
    Builder.CreateRetVoid();

    // Add this function to constructors.
    appendToGlobalCtors(M, Func, /*Priority*/ 1);
  }

  void createUnregisterFunction(GlobalVariable *FatbinDesc) {
    FunctionType *FuncTy =
        FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    Function *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                      "sycl.descriptor_unreg", &M);
    Func->setSection(".text.startup");

    // Get UnregFuncName function declaration.
    FunctionType *UnRegFuncTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C),
                          /*isVarArg=*/false);
    FunctionCallee UnRegFuncC =
        M.getOrInsertFunction("__sycl_unregister_lib", UnRegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(UnRegFuncC, FatbinDesc);
    Builder.CreateRetVoid();

    // Add this function to global destructors.
    appendToGlobalDtors(M, Func, /*Priority*/ 1);
  }

  void createSyclRegisterWithAtexitUnregister(GlobalVariable *FatbinDesc) {
    auto *UnregFuncTy =
        FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *UnregFunc =
        Function::Create(UnregFuncTy, GlobalValue::InternalLinkage,
                         "sycl.descriptor_unreg.atexit", &M);
    UnregFunc->setSection(".text.startup");

    // Declaration for __sycl_unregister_lib(void*).
    auto *UnregTargetTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C), false);
    FunctionCallee UnregTargetC =
        M.getOrInsertFunction("__sycl_unregister_lib", UnregTargetTy);

    // Body of the unregister wrapper.
    IRBuilder<> UnregBuilder(BasicBlock::Create(C, "entry", UnregFunc));
    UnregBuilder.CreateCall(UnregTargetC, FatbinDesc);
    UnregBuilder.CreateRetVoid();

    auto *RegFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *RegFunc = Function::Create(RegFuncTy, GlobalValue::InternalLinkage,
                                     "sycl.descriptor_reg", &M);
    RegFunc->setSection(".text.startup");

    auto *RegTargetTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C), false);
    FunctionCallee RegTargetC =
        M.getOrInsertFunction("__sycl_register_lib", RegTargetTy);

    // `atexit` takes a `void(*)()` function pointer arg and returns an i32.
    FunctionType *AtExitTy = FunctionType::get(
        Type::getInt32Ty(C), PointerType::getUnqual(C), false);
    FunctionCallee AtExitC = M.getOrInsertFunction("atexit", AtExitTy);

    IRBuilder<> RegBuilder(BasicBlock::Create(C, "entry", RegFunc));
    RegBuilder.CreateCall(RegTargetC, FatbinDesc);
    RegBuilder.CreateCall(AtExitC, UnregFunc);
    RegBuilder.CreateRetVoid();

    // Finally, add to global constructors.
    appendToGlobalCtors(M, RegFunc, /*Priority*/ 1);
  }

private:
  /// Creates structure corresponding to:
  /// \code
  ///  struct _pi_device_binary_property_struct {
  ///    char *Name;
  ///    void *ValAddr;
  ///    uint32_t Type;
  ///    uint64_t ValSize;
  ///  };
  /// \endcode
  StructType *getSyclPropTy() {
    return StructType::create({PointerType::getUnqual(C),
                               PointerType::getUnqual(C), Type::getInt32Ty(C),
                               Type::getInt64Ty(C)},
                              "_pi_device_binary_property_struct");
  }

  /// Creates a structure corresponding to:
  /// \code
  ///  struct _pi_device_binary_property_set_struct {
  ///    char *Name;
  ///    _pi_device_binary_property_struct* PropertiesBegin;
  ///    _pi_device_binary_property_struct* PropertiesEnd;
  ///  };
  /// \endcode
  StructType *getSyclPropSetTy() {
    return StructType::create({PointerType::getUnqual(C),
                               PointerType::getUnqual(C),
                               PointerType::getUnqual(C)},
                              "_pi_device_binary_property_set_struct");
  }

  IntegerType *getSizeTTy() {
    switch (M.getDataLayout().getPointerSize()) {
    case 4:
      return Type::getInt32Ty(C);
    case 8:
      return Type::getInt64Ty(C);
    }
    llvm_unreachable("unsupported pointer type size");
  }

  SmallVector<Constant *, 2> getSizetConstPair(size_t First, size_t Second) {
    IntegerType *SizeTTy = getSizeTTy();
    return SmallVector<Constant *, 2>{ConstantInt::get(SizeTTy, First),
                                      ConstantInt::get(SizeTTy, Second)};
  }

  /// Note: Properties aren't supported and the support is going
  /// to be added later.
  /// Creates a structure corresponding to:
  /// SYCL specific image descriptor type.
  /// \code
  /// struct __sycl.tgt_device_image {
  ///   // Version of this structure - for backward compatibility;
  ///   // all modifications which change order/type/offsets of existing fields
  ///   // should increment the version.
  ///   uint16_t Version;
  ///   // The kind of offload model the image employs.
  ///   uint8_t OffloadKind;
  ///   // Format of the image data - SPIRV, LLVMIR bitcode, etc.
  ///   uint8_t Format;
  ///   // Null-terminated string representation of the device's target
  ///   // architecture.
  ///   const char *Arch;
  ///   // A null-terminated string; target- and compiler-specific options
  ///   // which are passed to the device compiler at runtime.
  ///   const char *CompileOptions;
  ///   // A null-terminated string; target- and compiler-specific options
  ///   // which are passed to the device linker at runtime.
  ///   const char *LinkOptions;
  ///   // Pointer to the device binary image start.
  ///   void *ImageStart;
  ///   // Pointer to the device binary image end.
  ///   void *ImageEnd;
  ///   // The entry table.
  ///   __tgt_offload_entry *EntriesBegin;
  ///   __tgt_offload_entry *EntriesEnd;
  ///   _pi_device_binary_property_set_struct *PropertySetBegin;
  ///   _pi_device_binary_property_set_struct *PropertySetEnd;
  /// };
  /// \endcode
  StructType *getSyclDeviceImageTy() {
    return StructType::create(
        {
            Type::getInt16Ty(C),       // Version
            Type::getInt8Ty(C),        // OffloadKind
            Type::getInt8Ty(C),        // Format
            PointerType::getUnqual(C), // Arch
            PointerType::getUnqual(C), // CompileOptions
            PointerType::getUnqual(C), // LinkOptions
            PointerType::getUnqual(C), // ImageStart
            PointerType::getUnqual(C), // ImageEnd
            PointerType::getUnqual(C), // EntriesBegin
            PointerType::getUnqual(C), // EntriesEnd
            PointerType::getUnqual(C), // PropertySetBegin
            PointerType::getUnqual(C)  // PropertySetEnd
        },
        "__sycl.tgt_device_image");
  }

  /// Creates a structure for SYCL specific binary descriptor type. Corresponds
  /// to:
  ///
  /// \code
  ///  struct __sycl.tgt_bin_desc {
  ///    // version of this structure - for backward compatibility;
  ///    // all modifications which change order/type/offsets of existing fields
  ///    // should increment the version.
  ///    uint16_t Version;
  ///    uint16_t NumDeviceImages;
  ///    __sycl.tgt_device_image *DeviceImages;
  ///    // the offload entry table
  ///    __tgt_offload_entry *HostEntriesBegin;
  ///    __tgt_offload_entry *HostEntriesEnd;
  ///  };
  /// \endcode
  StructType *getSyclBinDescTy() {
    return StructType::create(
        {Type::getInt16Ty(C), Type::getInt16Ty(C), PointerType::getUnqual(C),
         PointerType::getUnqual(C), PointerType::getUnqual(C)},
        "__sycl.tgt_bin_desc");
  }

  Function *addDeclarationForNativeCPU(StringRef Name) {
    FunctionType *NativeCPUFuncTy = FunctionType::get(
        Type::getVoidTy(C),
        {PointerType::getUnqual(C), PointerType::getUnqual(C)}, false);
    FunctionType *NativeCPUBuiltinTy = FunctionType::get(
        PointerType::getUnqual(C), {PointerType::getUnqual(C)}, false);
    FunctionType *FTy;
    if (Name.starts_with("__dpcpp_nativecpu"))
      FTy = NativeCPUBuiltinTy;
    else
      FTy = NativeCPUFuncTy;
    auto FCalle = M.getOrInsertFunction(
        sycl::utils::addSYCLNativeCPUSuffix(Name).str(), FTy);
    Function *F = dyn_cast<Function>(FCalle.getCallee());
    if (F == nullptr)
      report_fatal_error("Unexpected callee");
    return F;
  }

  std::pair<Constant *, Constant *>
  addDeclarationsForNativeCPU(std::string Entries) {
    auto *NullPtr = llvm::ConstantPointerNull::get(PointerType::getUnqual(C));
    if (Entries.empty())
      return {NullPtr, NullPtr};

    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(Entries);
    // the Native CPU PI Plug-in expects the BinaryStart field to point to an
    // array of struct nativecpu_entry {
    //   char *kernelname;
    //   unsigned char *kernel_ptr;
    // };
    StructType *NCPUEntryT = StructType::create(
        {PointerType::getUnqual(C), PointerType::getUnqual(C)},
        "__nativecpu_entry");
    SmallVector<Constant *, 5> NativeCPUEntries;
    for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI) {
      auto *NewDecl = addDeclarationForNativeCPU(*LI);
      NativeCPUEntries.push_back(ConstantStruct::get(
          NCPUEntryT,
          {addStringToModule(*LI, "__ncpu_function_name"), NewDecl}));
    }

    // Add an empty entry that we use as end iterator
    auto *NativeCPUEndStr =
        addStringToModule("__nativecpu_end", "__ncpu_end_str");
    NativeCPUEntries.push_back(
        ConstantStruct::get(NCPUEntryT, {NativeCPUEndStr, NullPtr}));

    // Create the constant array containing the {kernel name, function pointers}
    // pairs
    ArrayType *ATy = ArrayType::get(NCPUEntryT, NativeCPUEntries.size());
    Constant *CA = ConstantArray::get(ATy, NativeCPUEntries);
    auto *GVar = new GlobalVariable(M, CA->getType(), true,
                                    GlobalVariable::InternalLinkage, CA,
                                    "__sycl_native_cpu_decls");
    auto *Begin = ConstantExpr::getGetElementPtr(GVar->getValueType(), GVar,
                                                 getSizetConstPair(0, 0));
    auto *End = ConstantExpr::getGetElementPtr(
        GVar->getValueType(), GVar,
        getSizetConstPair(0, NativeCPUEntries.size()));
    return std::make_pair(Begin, End);
  }  

  /// Adds a global readonly variable that is initialized by given
  /// \p Initializer to the module.
  GlobalVariable *addGlobalArrayVariable(const Twine &Name,
                                         ArrayRef<char> Initializer,
                                         const Twine &Section = "") {
    Constant *Arr = ConstantDataArray::get(M.getContext(), Initializer);
    GlobalVariable *Var =
        new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                           GlobalVariable::InternalLinkage, Arr, Name);
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    SmallVector<char, 32> NameBuf;
    StringRef SectionName = Section.toStringRef(NameBuf);
    if (!SectionName.empty())
      Var->setSection(SectionName);
    return Var;
  }

  /// Adds given \p Buf as a global variable into the module.
  /// \returns Pair of pointers that point at the beginning and the end of the
  /// variable.
  std::pair<Constant *, Constant *>
  addArrayToModule(ArrayRef<char> Buf, const Twine &Name,
                   const Twine &Section = "") {
    GlobalVariable *Var = addGlobalArrayVariable(Name, Buf, Section);
    Constant *ImageB = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                      getSizetConstPair(0, 0));
    Constant *ImageE = ConstantExpr::getGetElementPtr(
        Var->getValueType(), Var, getSizetConstPair(0, Buf.size()));
    return std::make_pair(ImageB, ImageE);
  }

  /// Adds given \p Data as constant byte array in the module.
  /// \returns Constant pointer to the added data. The pointer type does not
  /// carry size information.
  Constant *addRawDataToModule(ArrayRef<char> Data, const Twine &Name) {
    GlobalVariable *Var = addGlobalArrayVariable(Name, Data);
    Constant *DataPtr = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                       getSizetConstPair(0, 0));
    return DataPtr;
  }

  /// Creates a global variable of const char* type and creates an
  /// initializer that initializes it with \p Str.
  ///
  /// \returns Link-time constant pointer (constant expr) to that
  /// variable.
  Constant *addStringToModule(StringRef Str, const Twine &Name) {
    Constant *Arr = ConstantDataArray::getString(C, Str);
    GlobalVariable *Var =
        new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                           GlobalVariable::InternalLinkage, Arr, Name);
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    ConstantInt *Zero = ConstantInt::get(getSizeTTy(), 0);
    Constant *ZeroZero[] = {Zero, Zero};
    return ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroZero);
  }

  /// Creates a global variable of array of structs and initializes
  /// it with the given values in \p ArrayData.
  ///
  /// \returns Pair of Constants that point at array content.
  /// If \p ArrayData is empty then a returned pair contains nullptrs.
  std::pair<Constant *, Constant *>
  addStructArrayToModule(ArrayRef<Constant *> ArrayData, Type *ElemTy) {
    if (ArrayData.empty()) {
      auto *PtrTy = llvm::PointerType::getUnqual(ElemTy->getContext());
      auto *NullPtr = Constant::getNullValue(PtrTy);
      return std::make_pair(NullPtr, NullPtr);
    }

    assert(ElemTy == ArrayData[0]->getType() && "elem type mismatch");
    auto *Arr =
        ConstantArray::get(ArrayType::get(ElemTy, ArrayData.size()), ArrayData);
    auto *ArrGlob = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                       GlobalVariable::InternalLinkage, Arr,
                                       "__sycl_offload_prop_sets_arr");
    auto *ArrB = ConstantExpr::getGetElementPtr(
        ArrGlob->getValueType(), ArrGlob, getSizetConstPair(0, 0));
    auto *ArrE =
        ConstantExpr::getGetElementPtr(ArrGlob->getValueType(), ArrGlob,
                                       getSizetConstPair(0, ArrayData.size()));
    return std::pair<Constant *, Constant *>(ArrB, ArrE);
  }

  /// Creates a global variable that is initialized with the \p PropSet.
  ///
  /// \returns Pair of Constants that point at properties content.
  std::pair<Constant *, Constant *>
  addPropertySetToModule(const PropertySet &PropSet) {
    SmallVector<Constant *> PropInits;
    for (const auto &Prop : PropSet) {
      Constant *PropName = addStringToModule(Prop.first, "prop");
      Constant *PropValAddr = nullptr;
      Constant *PropType =
          ConstantInt::get(Type::getInt32Ty(C), Prop.second.index());
      Constant *PropValSize = nullptr;

      switch (Prop.second.index()) {
      case 0: {
        // for known scalar types ValAddr is null, ValSize keeps the value
        PropValAddr = Constant::getNullValue(PointerType::getUnqual(C));
        PropValSize =
            ConstantInt::get(Type::getInt64Ty(C), std::get<uint32_t>(Prop.second));
        break;
      }
      case 1: {
        const ByteArray &Arr = std::get<ByteArray>(Prop.second);
        PropValSize = ConstantInt::get(Type::getInt64Ty(C), Arr.size());
        PropValAddr = addRawDataToModule(ArrayRef<char>(Arr.begin(), Arr.size()), "prop_val");
        break;
      }
      default:
        llvm_unreachable_internal("unsupported property");
      }
      PropInits.push_back(ConstantStruct::get(SyclPropTy, PropName, PropValAddr,
                                              PropType, PropValSize));
    }
    return addStructArrayToModule(PropInits, SyclPropTy);
  }

  /// Creates a global variable that holds encoded given \p PropRegistry.
  /// In-object representation is demonstated below.
  ///
  /// column is a contiguous area of the wrapper object file;
  /// relative location of columns can be arbitrary
  ///
  /// \code
  ///                             _pi_device_binary_property_struct
  /// _pi_device_binary_property_set_struct   |
  ///                     |                   |
  ///                     v                   v
  /// ...             # ...                # ...
  /// PropSetsBegin--># Name0        +----># Name_00
  /// PropSetsEnd--+  # PropsBegin0--+     # ValAddr_00
  /// ...          |  # PropseEnd0------+  # Type_00
  ///              |  # Name1           |  # ValSize_00
  ///              |  # PropsBegin1---+ |  # ...
  ///              |  # PropseEnd1--+ | |  # Name_0n
  ///              +-># ...         | | |  # ValAddr_0n
  ///                 #             | | |  # Type_0n
  ///                 #             | | |  # ValSize_0n
  ///                 #             | | +-># ...
  ///                 #             | |    # ...
  ///                 #             | +---># Name_10
  ///                 #             |      # ValAddr_10
  ///                 #             |      # Type_10
  ///                 #             |      # ValSize_10
  ///                 #             |      # ...
  ///                 #             |      # Name_1m
  ///                 #             |      # ValAddr_1m
  ///                 #             |      # Type_1m
  ///                 #             |      # ValSize_1m
  ///                 #             +-----># ...
  ///                 #                    #
  /// \endcode
  ///
  /// \returns Pair of pointers to the beginning and end of the property set
  /// array, or a pair of nullptrs in case the properties file wasn't specified.
  std::pair<Constant *, Constant *>
  addPropertySetRegistry(const PropertySetRegistry &PropRegistry) {
    // transform all property sets to IR and get the middle column image into
    // the PropSetsInits
    SmallVector<Constant *> PropSetsInits;
    for (const auto &PropSet : PropRegistry) {
      // create content in the rightmost column and get begin/end pointers
      std::pair<Constant *, Constant *> Props =
          addPropertySetToModule(PropSet.second);
      // get the next the middle column element
      auto *Category = addStringToModule(PropSet.first, "SYCL_PropSetName");
      PropSetsInits.push_back(ConstantStruct::get(SyclPropSetTy, Category,
                                                  Props.first, Props.second));
    }
    // now get content for the leftmost column - create the top-level
    // PropertySetsBegin/PropertySetsBegin entries and return pointers to them
    return addStructArrayToModule(PropSetsInits, SyclPropSetTy);
  }

  /// Each image contains its own set of symbols, which may contain different
  /// symbols than other images. This function constructs an array of
  /// symbol entries for a particular image.
  ///
  /// \returns Pointers to the beginning and end of the array.
  std::pair<Constant *, Constant *>
  initOffloadEntriesPerImage(StringRef Entries, const Twine &OffloadKindTag) {
    SmallVector<Constant *> EntriesInits;
    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(
        Entries, /*BufferName*/ "", /*RequiresNullTerminator*/ false);
    for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI) {
      GlobalVariable *GV =
          emitOffloadingEntry(M, /*Kind*/ OffloadKind::OFK_SYCL,
                              Constant::getNullValue(PointerType::getUnqual(C)),
                              /*Name*/ *LI, /*Size*/ 0,
                              /*Flags*/ 0, /*Data*/ 0);
      EntriesInits.push_back(GV->getInitializer());
    }

    Constant *Arr = ConstantArray::get(
        ArrayType::get(EntryTy, EntriesInits.size()), EntriesInits);
    GlobalVariable *EntriesGV = new GlobalVariable(
        M, Arr->getType(), /*isConstant*/ true, GlobalVariable::InternalLinkage,
        Arr, OffloadKindTag + "entries_arr");

    Constant *EntriesB = ConstantExpr::getGetElementPtr(
        EntriesGV->getValueType(), EntriesGV, getSizetConstPair(0, 0));
    Constant *EntriesE = ConstantExpr::getGetElementPtr(
        EntriesGV->getValueType(), EntriesGV,
        getSizetConstPair(0, EntriesInits.size()));
    return std::make_pair(EntriesB, EntriesE);
  }

  /// Emits a global array that contains \p Address and \P Size. Also add
  /// it into llvm.used to force it to be emitted in the object file.
  void emitRegistrationFunctions(Constant *Address, size_t Size, Twine ImageID,
                                 StringRef OffloadKindTag) {
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(C);
    auto *ImgInfoArr =
        ConstantArray::get(ArrayType::get(IntPtrTy, 2),
                           {ConstantExpr::getPointerCast(Address, IntPtrTy),
                            ConstantInt::get(IntPtrTy, Size)});
    auto *ImgInfoVar = new GlobalVariable(
        M, ImgInfoArr->getType(), true, GlobalVariable::InternalLinkage,
        ImgInfoArr, Twine(OffloadKindTag) + ImageID + ".info");
    ImgInfoVar->setAlignment(
        MaybeAlign(M.getDataLayout().getTypeStoreSize(IntPtrTy) * 2u));
    ImgInfoVar->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    ImgInfoVar->setSection(".tgtimg");

    // Add image info to the used list to force it to be emitted to the
    // object.
    appendToUsed(M, ImgInfoVar);
  }

  Expected<Constant *> wrapImage(const OffloadBinary &OB, const Twine &ImageID,
                      StringRef OffloadKindTag) {
    // Note: Intel DPC++ compiler had 2 versions of this structure
    // and clang++ has a third different structure. To avoid ABI incompatibility
    // between generated device images the Version here starts from 3.
    constexpr uint16_t DeviceImageStructVersion = 3;
    Constant *Version =
        ConstantInt::get(Type::getInt16Ty(C), DeviceImageStructVersion);
    Constant *OffloadKindConstant = ConstantInt::get(
        Type::getInt8Ty(C), static_cast<uint8_t>(OB.getOffloadKind()));
    Constant *ImageKindConstant = ConstantInt::get(
        Type::getInt8Ty(C), static_cast<uint8_t>(OB.getImageKind()));
    StringRef Triple = OB.getString("triple");
    Constant *TripleConstant =
        addStringToModule(Triple, Twine(OffloadKindTag) + "target." + ImageID);
    Constant *CompileOptions =
        addStringToModule(Options.CompileOptions,
                          Twine(OffloadKindTag) + "opts.compile." + ImageID);
    Constant *LinkOptions = addStringToModule(
        Options.LinkOptions, Twine(OffloadKindTag) + "opts.link." + ImageID);

    // Note: NULL for now.
    //std::pair<Constant *, Constant *> PropertiesConstants = {
    //    Constant::getNullValue(PointerType::getUnqual(C)),
    //    Constant::getNullValue(PointerType::getUnqual(C))};
    //util::PropertySetRegistry::read(OB.getString("properties"));
    StringRef PropertiesStr = OB.getString("properties");
    if (PropertiesStr.empty())
      PropertiesStr = "{}"; // Empty JSON

    Expected<PropertySetRegistry> PropertiesOrErr = readPropertiesFromJSON(MemoryBufferRef(PropertiesStr, /*Identifier*/ ""));
    if (!PropertiesOrErr)
      return joinErrors(createStringError(formatv("failed to parse SYCL properties. content: {0}, ", OB.getString("properties"))), std::move(PropertiesOrErr.takeError()));
    std::pair<Constant *, Constant *> PropertiesConstants = addPropertySetRegistry(*PropertiesOrErr);

    StringRef RawImage = OB.getImage();
    std::pair<Constant *, Constant *> Binary = addArrayToModule(
        ArrayRef<char>(RawImage.begin(), RawImage.end()),
        Twine(OffloadKindTag) + ImageID + ".data", ".llvm.offloading");

    // For SYCL images offload entries are defined here per image.
    std::pair<Constant *, Constant *> ImageEntriesPtrs =
        initOffloadEntriesPerImage(OB.getString("symbols"), OffloadKindTag);

    // .first and .second arguments below correspond to start and end pointers
    // respectively.
    Constant *WrappedBinary = ConstantStruct::get(
        SyclDeviceImageTy, Version, OffloadKindConstant, ImageKindConstant,
        TripleConstant, CompileOptions, LinkOptions, Binary.first,
        Binary.second, ImageEntriesPtrs.first, ImageEntriesPtrs.second,
        PropertiesConstants.first, PropertiesConstants.second);

    // TODO: check that statement.
    if (true /*|| Options.EmitRegistrationFunctions*/)
      emitRegistrationFunctions(Binary.first, RawImage.size(),
                                ImageID, OffloadKindTag);

    return WrappedBinary;
  }

  GlobalVariable *combineWrappedImages(ArrayRef<Constant *> WrappedImages,
                                       StringRef OffloadKindTag) {
    Constant *ImagesData = ConstantArray::get(
        ArrayType::get(SyclDeviceImageTy, WrappedImages.size()), WrappedImages);
    GlobalVariable *ImagesGV =
        new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                           GlobalValue::InternalLinkage, ImagesData,
                           Twine(OffloadKindTag) + "device_images");
    ImagesGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    ConstantInt *Zero = ConstantInt::get(getSizeTTy(), 0);
    Constant *ZeroZero[] = {Zero, Zero};
    Constant *ImagesB = ConstantExpr::getGetElementPtr(ImagesGV->getValueType(),
                                                       ImagesGV, ZeroZero);

    Constant *EntriesB = Constant::getNullValue(PointerType::getUnqual(C));
    Constant *EntriesE = Constant::getNullValue(PointerType::getUnqual(C));
    static constexpr uint16_t BinDescStructVersion = 1;
    Constant *DescInit = ConstantStruct::get(
        SyclBinDescTy,
        ConstantInt::get(Type::getInt16Ty(C), BinDescStructVersion),
        ConstantInt::get(Type::getInt16Ty(C), WrappedImages.size()), ImagesB,
        EntriesB, EntriesE);

    return new GlobalVariable(M, DescInit->getType(), /*isConstant*/ true,
                              GlobalValue::InternalLinkage, DescInit,
                              Twine(OffloadKindTag) + "descriptor");
  }

  Module &M;
  LLVMContext &C;
  SYCLJITOptions Options;

  StructType *SyclPropTy = nullptr;
  StructType *SyclPropSetTy = nullptr;
  StructType *EntryTy = nullptr;
  StructType *SyclDeviceImageTy = nullptr;
  StructType *SyclBinDescTy = nullptr;
}; // end of SYCLWrapper

} // namespace

Error offloading::wrapOpenMPBinaries(Module &M, ArrayRef<ArrayRef<char>> Images,
                                     EntryArrayTy EntryArray,
                                     llvm::StringRef Suffix, bool Relocatable) {
  GlobalVariable *Desc =
      createBinDesc(M, Images, EntryArray, Suffix, Relocatable);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No binary descriptors created.");
  createRegisterFunction(M, Desc, Suffix);
  return Error::success();
}

Error offloading::wrapCudaBinary(Module &M, ArrayRef<char> Image,
                                 EntryArrayTy EntryArray,
                                 llvm::StringRef Suffix,
                                 bool EmitSurfacesAndTextures) {
  GlobalVariable *Desc = createFatbinDesc(M, Image, /*IsHip=*/false, Suffix);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No fatbin section created.");

  createRegisterFatbinFunction(M, Desc, /*IsHip=*/false, EntryArray, Suffix,
                               EmitSurfacesAndTextures);
  return Error::success();
}

Error offloading::wrapHIPBinary(Module &M, ArrayRef<char> Image,
                                EntryArrayTy EntryArray, llvm::StringRef Suffix,
                                bool EmitSurfacesAndTextures) {
  GlobalVariable *Desc = createFatbinDesc(M, Image, /*IsHip=*/true, Suffix);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No fatbin section created.");

  createRegisterFatbinFunction(M, Desc, /*IsHip=*/true, EntryArray, Suffix,
                               EmitSurfacesAndTextures);
  return Error::success();
}

Error llvm::offloading::wrapSYCLBinaries(llvm::Module &M, ArrayRef<char> Buffer,
                                         SYCLJITOptions Options) {
  SYCLWrapper W(M, Options);
  MemoryBufferRef MBR(StringRef(Buffer.begin(), Buffer.size()),
                      /*Identifier*/ "");
  SmallVector<OffloadFile> OffloadFiles;
  if (Error E = extractOffloadBinaries(MBR, OffloadFiles))
    return E;

  Expected<GlobalVariable *> DescOrErr = W.createFatbinDesc(OffloadFiles);
  if (!DescOrErr)
    return joinErrors(createStringError("failed to wrap SYCL Binaries: "), std::move(DescOrErr.takeError()));

  if (Triple(M.getTargetTriple()).isOSWindows()) {
    W.createSyclRegisterWithAtexitUnregister(*DescOrErr);
  } else {
    W.createRegisterFatbinFunction(*DescOrErr);
    W.createUnregisterFunction(*DescOrErr);
  }
  return Error::success();
}
