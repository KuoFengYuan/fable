//
//  fable-Bridging-Header.h
//  fable — 把 msplat 的 C 訓練 API 曝露給 Swift。
//
//  msplat（Metal 版 3D Gaussian Splatting 訓練器，vendored 於 Training/msplat/，
//  Apache-2.0）以純 C API 對外，Swift 透過本 bridging header 直接呼叫。
//

#import "msplat_c_api.h"
