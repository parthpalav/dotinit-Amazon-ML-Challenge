// Same posting limits and rule bits as legacy lookup, reordered for locality.
#include <algorithm>
#include <cstdint>
#include <vector>
#include <unordered_map>
#pragma pack(push,1)
struct Entry { uint64_t key; uint32_t row; };
#pragma pack(pop)
struct Query { uint64_t key; uint32_t anchor; uint16_t slot; };
extern "C" size_t sorted_lookup(const Entry* data,size_t size,const uint64_t* keys,size_t n,
 int width,uint32_t maxpost,uint32_t* rows,uint16_t* masks,uint32_t* offsets,size_t capacity) {
 std::vector<Query> queries;queries.reserve(n*width);
 for(size_t i=0;i<n;++i)for(int j=0;j<width;++j)if(keys[i*width+j])queries.push_back({keys[i*width+j],uint32_t(i),uint16_t(j)});
 std::sort(queries.begin(),queries.end(),[](const Query&a,const Query&b){return a.key<b.key;});
 std::vector<std::unordered_map<uint32_t,uint16_t>> found(n);
 for(auto&q:queries) {
  auto lo=std::lower_bound(data,data+size,q.key,[](const Entry&a,uint64_t k){return a.key<k;});
  auto hi=std::upper_bound(lo,data+size,q.key,[](uint64_t k,const Entry&a){return k<a.key;});
  int j=q.slot;size_t limit=(width==24&&(j==0||j==21))?maxpost*5:maxpost;
  if(size_t(hi-lo)>limit)continue;
  uint16_t mask=width==64?(j<56?128:256):(j==0?32:j<4?1:j<12?8:j<20?16:j==20?4:j==21?16:64);
  for(auto p=lo;p!=hi;++p)found[q.anchor][p->row]|=mask;
 }
 size_t used=0;offsets[0]=0;
 for(size_t i=0;i<n;++i) {
  std::vector<std::pair<uint32_t,uint16_t>> a(found[i].begin(),found[i].end());std::sort(a.begin(),a.end());
  if(used+a.size()>capacity)return SIZE_MAX;
  for(auto&x:a){rows[used]=x.first;masks[used++]=x.second;}offsets[i+1]=used;
 }
 return used;
}
