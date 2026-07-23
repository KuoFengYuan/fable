#ifndef METAL_TENSOR_H
#define METAL_TENSOR_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <utility>
#include <CoreFoundation/CoreFoundation.h>  // CFRetain/CFRelease（純 C API，.cpp/.mm 皆可）

// Forward-declare the Metal buffer type for C++ compatibility.
// Full Metal/Metal.h is only needed in .mm files.
#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* MTLBufferRef;  // opaque handle in pure C++
#endif

enum class DType : uint8_t {
    Float32,
    Int32,
    Int64,
    UInt8,
    Float64,
};

inline size_t dtypeSize(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::Float64: return 8;
    }
}

// Lightweight GPU tensor — wraps an MTLBuffer with shape metadata.
class MTensor {
public:
    MTensor() = default;

#ifdef __OBJC__
    // GPU allocation (Objective-C++ only)
    MTensor(id<MTLDevice> device, std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        if (bytes == 0) bytes = 4;
        id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        _buffer = (__bridge_retained void*)buf;
        _data = [buf contents];  // cache CPU-accessible pointer for C++ access
    }

    id<MTLBuffer> buffer() const { return (__bridge id<MTLBuffer>)_buffer; }
#endif

    // CPU allocation (no Metal buffer)
    MTensor(std::vector<int64_t> shape, DType dtype)
        : _shape(std::move(shape)), _dtype(dtype) {
        _numel = 1;
        for (auto s : _shape) _numel *= s;
        size_t bytes = _numel * dtypeSize(_dtype);
        _cpu_data.resize(bytes);
    }

    // ── 值語意（RAII 引用計數）──
    // 舊版用編譯器預設淺拷貝（複製 _buffer 指標卻不 retain）＋無解構子＋手動 reset()：
    //   → 兩個 MTensor 共用同一 buffer 時各自 reset() = CFRelease 兩次（雙重釋放當機）；
    //   → `x = 新tensor` 覆寫舊指標卻不釋放（記憶體洩漏）。
    // 改為正確引用計數：拷貝/資料檢視 CFRetain、解構/覆寫 CFRelease → 杜絕雙重釋放與洩漏。
    // 用 CoreFoundation C API（非 __OBJC__ 條件式），避免 inline reset() 在 .cpp/.mm 兩種定義的 ODR 分歧。
    MTensor(const MTensor& o)
        : _buffer(o._buffer), _data(o._data), _cpu_data(o._cpu_data),
          _shape(o._shape), _dtype(o._dtype), _numel(o._numel) {
        if (_buffer) CFRetain((CFTypeRef)_buffer);
    }
    MTensor& operator=(const MTensor& o) {
        if (this != &o) {
            if (o._buffer) CFRetain((CFTypeRef)o._buffer);
            if (_buffer) CFRelease((CFTypeRef)_buffer);
            _buffer = o._buffer; _data = o._data; _cpu_data = o._cpu_data;
            _shape = o._shape; _dtype = o._dtype; _numel = o._numel;
        }
        return *this;
    }
    MTensor(MTensor&& o) noexcept
        : _buffer(o._buffer), _data(o._data), _cpu_data(std::move(o._cpu_data)),
          _shape(std::move(o._shape)), _dtype(o._dtype), _numel(o._numel) {
        o._buffer = nullptr; o._data = nullptr; o._numel = 0;
    }
    MTensor& operator=(MTensor&& o) noexcept {
        if (this != &o) {
            if (_buffer) CFRelease((CFTypeRef)_buffer);
            _buffer = o._buffer; _data = o._data; _cpu_data = std::move(o._cpu_data);
            _shape = std::move(o._shape); _dtype = o._dtype; _numel = o._numel;
            o._buffer = nullptr; o._data = nullptr; o._numel = 0;
        }
        return *this;
    }
    ~MTensor() { reset(); }

    bool defined() const { return _buffer != nullptr || !_cpu_data.empty(); }
    bool isGpu() const { return _buffer != nullptr; }

    int64_t numel() const { return _numel; }
    int64_t size(int dim) const {
        if (dim < 0) dim += _shape.size();
        return _shape[dim];
    }
    int ndim() const { return (int)_shape.size(); }
    const std::vector<int64_t>& shape() const { return _shape; }
    DType dtype() const { return _dtype; }
    size_t elementSize() const { return dtypeSize(_dtype); }
    size_t nbytes() const { return _numel * dtypeSize(_dtype); }

    void* data_ptr() {
        if (_data) return _data;
        return _cpu_data.data();
    }
    const void* data_ptr() const {
        if (_data) return _data;
        return _cpu_data.data();
    }

    template<typename T> T* data() { return static_cast<T*>(data_ptr()); }
    template<typename T> const T* data() const { return static_cast<const T*>(data_ptr()); }

    void zero() {
        memset(data_ptr(), 0, _numel * dtypeSize(_dtype));
    }

    // Create a CPU copy of the data
    MTensor cpu() const {
        MTensor out(_shape, _dtype);
        memcpy(out.data_ptr(), data_ptr(), nbytes());
        return out;
    }

    void reset() {
        if (_buffer) { CFRelease((CFTypeRef)_buffer); }
        _buffer = nullptr;
        _data = nullptr;
        _cpu_data.clear();
        _shape.clear();
        _numel = 0;
    }

    // Stride for dim 0 (elements per row)
    int64_t stride0() const {
        if (_shape.size() <= 1) return 1;
        int64_t s = 1;
        for (size_t i = 1; i < _shape.size(); i++) s *= _shape[i];
        return s;
    }

    // Create a view of the first `n` elements along dim 0.
    // 共有底層 MTLBuffer 並「retain」→ 為擁有者之一，靠引用計數保命（不再有 parent 先亡的 UAF）。
    MTensor view(int64_t n) const {
        MTensor v;
        v._buffer = _buffer;  // 共有 buffer
        if (v._buffer) CFRetain((CFTypeRef)v._buffer);
        v._data = _data;      // 共有 CPU 可存取指標（buffer 由引用計數保活）
        v._shape = _shape;
        v._shape[0] = n;
        v._dtype = _dtype;
        v._numel = n * stride0();
        return v;
    }

private:
    void* _buffer = nullptr;  // retained id<MTLBuffer> as void*
    void* _data = nullptr;    // cached CPU-accessible pointer (shared memory on Apple Silicon)
    std::vector<uint8_t> _cpu_data;
    std::vector<int64_t> _shape;
    DType _dtype = DType::Float32;
    int64_t _numel = 0;
};

// Factory helpers (Objective-C++ only)
#ifdef __OBJC__
inline MTensor mtensor_empty(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    return MTensor(dev, std::move(shape), dt);
}

inline MTensor mtensor_zeros(id<MTLDevice> dev, std::vector<int64_t> shape, DType dt) {
    MTensor t(dev, std::move(shape), dt);
    t.zero();
    return t;
}
#endif

#endif
