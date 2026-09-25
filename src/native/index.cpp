// Compact, deterministic retrieval acceleration. No external data or libraries.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#pragma pack(push,1)
struct Entry { uint64_t key; uint32_t row; };
#pragma pack(pop)
static const int WIDTH=24;
static uint64_t hashstr(const std::string& s) {
 uint64_t h=1469598103934665603ULL;
 for(unsigned char c:s) {h^=c;h*=1099511628211ULL;} return h ? h : 1;
}
static uint64_t mix(uint64_t x) {
 x^=x>>30; x*=0xbf58476d1ce4e5b9ULL; x^=x>>27; x*=0x94d049bb133111ebULL; return x^(x>>31);
}
static std::vector<std::string> tokens(const std::string& s) {
 std::vector<std::string> out; size_t pos=0;
 while(pos<s.size()) {size_t end=s.find(' ',pos); if(end==std::string::npos)end=s.size(); if(end>pos)out.push_back(s.substr(pos,end-pos));pos=end+1;} return out;
}
static bool generic(const std::string& s) {
 static const std::string words="|and|the|of|company|corporation|group|services|international|inc|ltd|limited|private|pvt|llc|llp|corp|co|";
 return words.find("|"+s+"|")!=std::string::npos;
}
static void lsh(const std::string& text, int field, uint64_t* out) {
 if(text.empty())return;
 std::string compact; for(char c:text)if(c!=' ')compact+=c;
 if(compact.empty())return;
 uint64_t minima[16];std::fill(minima,minima+16,UINT64_MAX);
 size_t width=std::min(size_t(4),compact.size());
 for(size_t i=0;i+width<=compact.size();++i) {
  uint64_t h=hashstr(compact.substr(i,width));
  for(int j=0;j<16;++j)minima[j]=std::min(minima[j],mix(h+uint64_t(j+1)*0x9e3779b97f4a7c15ULL));
 }
 for(int j=0;j<8;++j) out[j]=mix(minima[2*j]^(minima[2*j+1]<<1)^uint64_t(field*16+j+1)*0x517cc1b727220a95ULL)|1ULL;
}
extern "C" void ber_keys(const char** names,const char** addresses,const char** countries,size_t n,uint64_t* out) {
 for(size_t i=0;i<n;++i) {
  auto k=out+i*WIDTH; std::fill(k,k+WIDTH,0);
  std::string name(names[i]),address(addresses[i]),country(countries[i]);
  auto words=tokens(name); std::vector<std::string> informative;
  for(auto& t:words)if(t.size()>1&&!generic(t))informative.push_back(t);
  std::sort(informative.begin(),informative.end(),[](const auto&a,const auto&b){return a.size()!=b.size()?a.size()>b.size():a<b;});
  informative.erase(std::unique(informative.begin(),informative.end()),informative.end());
  if(!name.empty())k[0]=hashstr("N|"+name);
  for(size_t j=0;j<std::min(size_t(3),informative.size());++j)if(!country.empty())k[1+j]=hashstr("T|"+country+"|"+informative[j]);
  lsh(name,1,k+4);lsh(address,2,k+12);
  auto aw=tokens(address);std::string postal,number;
  for(auto&t:aw)if(!t.empty()&&std::all_of(t.begin(),t.end(),[](unsigned char c){return c>='0'&&c<='9';})) {
   if(number.empty())number=t;
   if(t.size()>=4&&t.size()<=10)postal=t;
  }
  if(!postal.empty()&&!country.empty())k[20]=hashstr("P|"+country+"|"+postal);
  if(!address.empty()) {std::sort(aw.begin(),aw.end());std::string bag;for(auto&t:aw)bag+=t+" ";k[21]=hashstr("A|"+bag);}
  if(!number.empty()&&!country.empty())for(size_t j=0;j<std::min(size_t(2),informative.size());++j)k[22+j]=hashstr("X|"+country+"|"+number+"|"+informative[j]);
 }
}
struct Index {int fd;size_t size;Entry* data;};
extern "C" void* ber_open(const char* path) {
 int fd=open(path,O_RDONLY);if(fd<0)return nullptr;struct stat s;if(fstat(fd,&s)!=0){close(fd);return nullptr;}
 if(s.st_size==0)return new Index{fd,0,nullptr};
 void*p=mmap(nullptr,s.st_size,PROT_READ,MAP_SHARED,fd,0);if(p==MAP_FAILED){close(fd);return nullptr;}
 return new Index{fd,size_t(s.st_size)/sizeof(Entry),(Entry*)p};
}
extern "C" void ber_close(void* ptr) {if(!ptr)return;auto x=(Index*)ptr;if(x->data)munmap(x->data,x->size*sizeof(Entry));close(x->fd);delete x;}
extern "C" size_t ber_lookup(void* ptr,const uint64_t* keys,size_t n,uint32_t maxpost,uint32_t* rows,uint16_t* masks,uint32_t* offsets,size_t capacity) {
 auto x=(Index*)ptr;size_t used=0;offsets[0]=0;
 for(size_t i=0;i<n;++i) {
  std::unordered_map<uint32_t,uint16_t> found;
  for(int j=0;j<WIDTH;++j) {
   uint64_t key=keys[i*WIDTH+j];if(!key||!x->size)continue;
   auto lo=std::lower_bound(x->data,x->data+x->size,key,[](const Entry&a,uint64_t b){return a.key<b;});
   auto hi=std::upper_bound(lo,x->data+x->size,key,[](uint64_t a,const Entry&b){return a<b.key;});
   size_t limit=(j==0||j==21)?maxpost*5:maxpost;
   if(size_t(hi-lo)>limit)continue;
   uint16_t mask=j==0?32:j<4?1:j<12?8:j<20?16:j==20?4:j==21?16:64;
   for(auto p=lo;p!=hi;++p)found[p->row]|=mask;
  }
  std::vector<std::pair<uint32_t,uint16_t>> ordered(found.begin(),found.end());std::sort(ordered.begin(),ordered.end());
  if(used+ordered.size()>capacity)return SIZE_MAX;
  for(auto& v:ordered){rows[used]=v.first;masks[used]=v.second;++used;}offsets[i+1]=used;
 }
 return used;
}
