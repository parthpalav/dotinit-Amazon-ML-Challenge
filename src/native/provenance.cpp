#include <algorithm>
#include <cstddef>
#include <cstdint>
extern "C" void ber_pair_masks(const uint64_t* anchors,const uint64_t* targets,const uint32_t* ai,const uint32_t* ti,size_t n,size_t width,uint16_t* result) {
 for(size_t i=0;i<n;++i) {
  const uint64_t* a=anchors+size_t(ai[i])*width;const uint64_t* t=targets+size_t(ti[i])*width;uint16_t mask=0;
  for(size_t j=0;j<width;++j)if(a[j]&&std::binary_search(t,t+width,a[j])) {
   mask|=width==64?(j<56?128:256):(j==0?32:j<4?1:j<12?8:j<20?16:j==20?4:j==21?16:64);
  }
  result[i]|=mask;
 }
}
