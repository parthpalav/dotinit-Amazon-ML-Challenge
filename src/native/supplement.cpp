// Supplemental number/address-token and unordered name-token-pair retrieval.
#include "index.cpp"
static const int EXTRA_WIDTH=64;
extern "C" void ber_extra_keys(const char** names,const char** addresses,const char** countries,size_t n,uint64_t* out) {
 for(size_t i=0;i<n;++i) {
  auto k=out+i*EXTRA_WIDTH;std::fill(k,k+EXTRA_WIDTH,0);
  std::string country(countries[i]);if(country.empty())continue;
  auto nw=tokens(names[i]),aw=tokens(addresses[i]);std::vector<std::string> numbers,words,informative;
  for(auto&t:aw) {
   for(size_t p=0;p<t.size();) {
    if(t[p]>='0'&&t[p]<='9'){size_t e=p+1;while(e<t.size()&&t[e]>='0'&&t[e]<='9')++e;numbers.push_back(t.substr(p,e-p));p=e;}else ++p;
   }
   if(t.size()>=3&&std::any_of(t.begin(),t.end(),[](unsigned char c){return (c>='a'&&c<='z')||c>=128;}))words.push_back(t);
  }
  std::sort(numbers.begin(),numbers.end());numbers.erase(std::unique(numbers.begin(),numbers.end()),numbers.end());
  std::sort(words.begin(),words.end());words.erase(std::unique(words.begin(),words.end()),words.end());
  std::vector<uint64_t> address_keys;
  for(auto&number:numbers)for(auto&word:words)address_keys.push_back(hashstr("AN|"+country+"|"+number+"|"+word));
  std::sort(address_keys.begin(),address_keys.end());
  // Deterministic bottom-k only for unusually long addresses; average record uses far fewer keys.
  for(size_t j=0;j<std::min(size_t(56),address_keys.size());++j)k[j]=address_keys[j];
  for(auto&t:nw)if(t.size()>1&&!generic(t))informative.push_back(t);
  std::sort(informative.begin(),informative.end());informative.erase(std::unique(informative.begin(),informative.end()),informative.end());
  std::vector<uint64_t> name_keys;
  for(size_t a=0;a<informative.size();++a)for(size_t b=a+1;b<informative.size();++b)name_keys.push_back(hashstr("NP|"+country+"|"+informative[a]+"|"+informative[b]));
  std::sort(name_keys.begin(),name_keys.end());for(size_t j=0;j<std::min(size_t(8),name_keys.size());++j)k[56+j]=name_keys[j];
 }
}
extern "C" size_t ber_extra_lookup(void* ptr,const uint64_t* keys,size_t n,uint32_t maxpost,uint32_t* rows,uint16_t* masks,uint32_t* offsets,size_t capacity) {
 auto x=(Index*)ptr;size_t used=0;offsets[0]=0;
 for(size_t i=0;i<n;++i) {
  std::unordered_map<uint32_t,uint16_t> found;
  for(int j=0;j<EXTRA_WIDTH;++j) {
   uint64_t key=keys[i*EXTRA_WIDTH+j];if(!key||!x->size)continue;
   auto lo=std::lower_bound(x->data,x->data+x->size,key,[](const Entry&a,uint64_t b){return a.key<b;});
   auto hi=std::upper_bound(lo,x->data+x->size,key,[](uint64_t a,const Entry&b){return a<b.key;});
   if(size_t(hi-lo)>maxpost)continue;
   for(auto p=lo;p!=hi;++p)found[p->row]|=(j<56?128:256);
  }
  std::vector<std::pair<uint32_t,uint16_t>> ordered(found.begin(),found.end());std::sort(ordered.begin(),ordered.end());
  if(used+ordered.size()>capacity)return SIZE_MAX;
  for(auto&v:ordered){rows[used]=v.first;masks[used]=v.second;++used;}offsets[i+1]=used;
 }
 return used;
}
extern "C" size_t ber_merge(const uint32_t* ar,const uint16_t* am,const uint32_t* ao,
 const uint32_t* br,const uint16_t* bm,const uint32_t* bo,size_t n,uint32_t* rows,uint16_t* masks,uint32_t* offsets) {
 size_t used=0;offsets[0]=0;
 for(size_t i=0;i<n;++i){size_t a=ao[i],b=bo[i];
  while(a<ao[i+1]||b<bo[i+1]){
   if(b==bo[i+1]||(a<ao[i+1]&&ar[a]<br[b])){rows[used]=ar[a];masks[used++]=am[a++];}
   else if(a==ao[i+1]||br[b]<ar[a]){rows[used]=br[b];masks[used++]=bm[b++];}
   else{rows[used]=ar[a];masks[used++]=am[a++]|bm[b++];}
  }offsets[i+1]=used;
 }return used;
}
