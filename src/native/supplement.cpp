// Supplemental number/address-token, Indic transliteration + Soundex, and unordered name-token-pair retrieval.
#include "index.cpp"

static const int EXTRA_WIDTH = 64;

static char transliterate_indic_cp(uint32_t cp) {
    if (cp < 0x0900 || cp > 0x0D7F) return 0;
    uint32_t off = (cp - 0x0900) % 0x80;
    switch (off) {
        case 0x15: case 0x16: case 0x17: case 0x18: return 'k';
        case 0x19: return 'n';
        case 0x1A: case 0x1B: case 0x1C: case 0x1D: return 'c';
        case 0x1E: return 'n';
        case 0x1F: case 0x20: case 0x21: case 0x22: case 0x24: case 0x25: case 0x26: case 0x27: return 't';
        case 0x23: case 0x28: return 'n';
        case 0x2A: case 0x2B: case 0x2C: case 0x2D: return 'p';
        case 0x2E: return 'm';
        case 0x2F: return 'y';
        case 0x30: return 'r';
        case 0x32: return 'l';
        case 0x35: return 'v';
        case 0x36: case 0x37: case 0x38: return 's';
        case 0x39: return 'h';
        case 0x05: case 0x06: case 0x3E: return 'a';
        case 0x07: case 0x08: case 0x3F: case 0x40: return 'i';
        case 0x09: case 0x0A: case 0x41: case 0x42: return 'u';
        case 0x0F: case 0x10: case 0x47: case 0x48: return 'e';
        case 0x13: case 0x14: case 0x4B: case 0x4C: return 'o';
        default: return 0;
    }
}

static std::string transliterate_indic_utf8(const std::string& s) {
    std::string out;
    for (size_t p = 0; p < s.size();) {
        unsigned char c = s[p];
        if ((c & 0x80) == 0) {
            out.push_back(c);
            p++;
        } else if ((c & 0xE0) == 0xE0 && p + 2 < s.size()) {
            uint32_t cp = ((uint32_t(c & 0x0F) << 12) | (uint32_t(s[p+1] & 0x3F) << 6) | uint32_t(s[p+2] & 0x3F));
            char t = transliterate_indic_cp(cp);
            if (t) out.push_back(t);
            p += 3;
        } else if ((c & 0xE0) == 0xC0 && p + 1 < s.size()) {
            uint32_t cp = ((uint32_t(c & 0x1F) << 6) | uint32_t(s[p+1] & 0x3F));
            if ((cp >= 0xC0 && cp <= 0xC5) || (cp >= 0xE0 && cp <= 0xE5)) out.push_back('a');
            else if ((cp >= 0xC8 && cp <= 0xCB) || (cp >= 0xE8 && cp <= 0xEB)) out.push_back('e');
            else if ((cp >= 0xCC && cp <= 0xCF) || (cp >= 0xEC && cp <= 0xEF)) out.push_back('i');
            else if ((cp >= 0xD2 && cp <= 0xD6) || (cp >= 0xF2 && cp <= 0xF6)) out.push_back('o');
            else if ((cp >= 0xD9 && cp <= 0xDC) || (cp >= 0xF9 && cp <= 0xFC)) out.push_back('u');
            else if (cp == 0xC7 || cp == 0xE7) out.push_back('c');
            p += 2;
        } else {
            p++;
        }
    }
    return out;
}

static std::string soundex_calc(const std::string& s) {
    if (s.empty()) return "";
    char first = toupper(s[0]);
    auto code = [](char c) -> char {
        c = toupper(c);
        if (c=='B'||c=='F'||c=='P'||c=='V') return '1';
        if (c=='C'||c=='G'||c=='J'||c=='K'||c=='Q'||c=='S'||c=='X'||c=='Z') return '2';
        if (c=='D'||c=='T') return '3';
        if (c=='L') return '4';
        if (c=='M'||c=='N') return '5';
        if (c=='R') return '6';
        return '0';
    };
    std::string tail;
    char prev = code(first);
    for (size_t i = 1; i < s.size(); ++i) {
        char cd = code(s[i]);
        if (cd != '0' && cd != prev) tail += cd;
        prev = cd;
    }
    std::string res = std::string(1, first) + tail + "0000";
    return res.substr(0, 4);
}

static bool generic_sx(const std::string& sx) {
    return sx == "P613" || sx == "L533" || sx == "S612" || sx == "C515";
}

static bool generic_addr(const std::string& s) {
    static const std::string words = "|rd|st|ave|ln|blvd|dr|near|main|building|apt|ste|and|the|floor|block|flat|opposite|unit|pmb|road|street|lane|avenue|drive|";
    return words.find("|" + s + "|") != std::string::npos;
}

extern "C" void ber_extra_keys(const char** names, const char** addresses, const char** countries, size_t n, uint64_t* out) {
    for (size_t i = 0; i < n; ++i) {
        auto k = out + i * EXTRA_WIDTH;
        std::fill(k, k + EXTRA_WIDTH, 0);
        std::string country(countries[i]);
        if (country.empty()) continue;

        std::string name_raw(names[i]);
        std::string name_trans = transliterate_indic_utf8(name_raw);
        auto nw_raw = tokens(name_raw);
        auto nw_trans = tokens(name_trans);
        auto aw = tokens(addresses[i]);

        std::vector<std::string> numbers, words, informative_name;
        for (auto& t : aw) {
            for (size_t p = 0; p < t.size();) {
                if (t[p] >= '0' && t[p] <= '9') {
                    size_t e = p + 1;
                    while (e < t.size() && t[e] >= '0' && t[e] <= '9') ++e;
                    numbers.push_back(t.substr(p, e - p));
                    p = e;
                } else ++p;
            }
            if (t.size() >= 3 && !generic_addr(t) && !std::all_of(t.begin(), t.end(), [](char c){ return c >= '0' && c <= '9'; })) {
                words.push_back(t);
            }
        }
        std::sort(numbers.begin(), numbers.end());
        numbers.erase(std::unique(numbers.begin(), numbers.end()), numbers.end());
        std::sort(words.begin(), words.end());
        words.erase(std::unique(words.begin(), words.end()), words.end());

        // Informative name tokens (raw)
        for (auto& t : nw_raw) {
            if (t.size() > 1 && !generic(t)) informative_name.push_back(t);
        }
        std::sort(informative_name.begin(), informative_name.end());
        informative_name.erase(std::unique(informative_name.begin(), informative_name.end()), informative_name.end());

        // Phonetic Soundex from transliterated name
        std::vector<std::string> sxs;
        for (auto& t : nw_trans) {
            if (t.size() > 2 && !generic(t)) {
                std::string sx = soundex_calc(t);
                if (!sx.empty() && !generic_sx(sx)) sxs.push_back(sx);
            }
        }
        std::sort(sxs.begin(), sxs.end());
        sxs.erase(std::unique(sxs.begin(), sxs.end()), sxs.end());

        size_t idx = 0;

        // 1. Channel 1a: SXN (Soundex + Address Number) -> up to 10 keys
        std::vector<uint64_t> sxn_keys;
        for (auto& sx : sxs) {
            for (auto& num : numbers) {
                sxn_keys.push_back(hashstr("SXN|" + country + "|" + sx + "|" + num));
            }
        }
        std::sort(sxn_keys.begin(), sxn_keys.end());
        for (size_t j = 0; j < std::min(size_t(10), sxn_keys.size()) && idx < 10; ++j) {
            k[idx++] = sxn_keys[j];
        }
        idx = 10;

        // 2. Channel 1b: SXW (Soundex + Address Word) -> up to 14 keys
        std::vector<uint64_t> sxw_keys;
        for (auto& sx : sxs) {
            for (auto& w : words) {
                sxw_keys.push_back(hashstr("SXW|" + country + "|" + sx + "|" + w));
            }
        }
        std::sort(sxw_keys.begin(), sxw_keys.end());
        for (size_t j = 0; j < std::min(size_t(14), sxw_keys.size()) && idx < 24; ++j) {
            k[idx++] = sxw_keys[j];
        }
        idx = 24;

        // 3. Channel 2a: AN (Address Number + Word) -> up to 16 keys
        std::vector<uint64_t> an_keys;
        for (auto& num : numbers) {
            for (auto& w : words) {
                an_keys.push_back(hashstr("AN|" + country + "|" + num + "|" + w));
            }
        }
        std::sort(an_keys.begin(), an_keys.end());
        for (size_t j = 0; j < std::min(size_t(16), an_keys.size()) && idx < 40; ++j) {
            k[idx++] = an_keys[j];
        }
        idx = 40;

        // 4. Channel 2b: AW2 (Address Word Pairs) -> up to 14 keys
        std::vector<uint64_t> aw2_keys;
        for (size_t a = 0; a < words.size(); ++a) {
            for (size_t b = a + 1; b < words.size(); ++b) {
                aw2_keys.push_back(hashstr("AW2|" + country + "|" + words[a] + "|" + words[b]));
            }
        }
        std::sort(aw2_keys.begin(), aw2_keys.end());
        for (size_t j = 0; j < std::min(size_t(14), aw2_keys.size()) && idx < 54; ++j) {
            k[idx++] = aw2_keys[j];
        }
        idx = 54;

        // 5. Channel 3: NP (Name word pairs) -> up to 10 keys
        std::vector<uint64_t> np_keys;
        for (size_t a = 0; a < informative_name.size(); ++a) {
            for (size_t b = a + 1; b < informative_name.size(); ++b) {
                np_keys.push_back(hashstr("NP|" + country + "|" + informative_name[a] + "|" + informative_name[b]));
            }
        }
        std::sort(np_keys.begin(), np_keys.end());
        for (size_t j = 0; j < std::min(size_t(10), np_keys.size()) && idx < 64; ++j) {
            k[idx++] = np_keys[j];
        }
    }
}

extern "C" size_t ber_extra_lookup(void* ptr, const uint64_t* keys, size_t n, uint32_t maxpost, uint32_t* rows, uint16_t* masks, uint32_t* offsets, size_t capacity) {
    auto x = (Index*)ptr;
    size_t used = 0;
    offsets[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        std::unordered_map<uint32_t, uint16_t> found;
        for (int j = 0; j < EXTRA_WIDTH; ++j) {
            uint64_t key = keys[i * EXTRA_WIDTH + j];
            if (!key || !x->size) continue;
            auto lo = std::lower_bound(x->data, x->data + x->size, key, [](const Entry& a, uint64_t b) { return a.key < b; });
            auto hi = std::upper_bound(lo, x->data + x->size, key, [](uint64_t a, const Entry& b) { return a < b.key; });
            if (size_t(hi - lo) > maxpost) continue;
            // Rule masks:
            // j < 10: SXN -> mask 64 (name_number)
            // 10 <= j < 24: SXW -> mask 128 (address_number)
            // 24 <= j < 40: AN  -> mask 128 (address_number)
            // 40 <= j < 54: AW2 -> mask 128 (address_number)
            // 54 <= j < 64: NP  -> mask 256 (name_pair)
            uint16_t mask = (j < 10) ? 64 : (j < 54) ? 128 : 256;
            for (auto p = lo; p != hi; ++p) found[p->row] |= mask;
        }
        std::vector<std::pair<uint32_t, uint16_t>> ordered(found.begin(), found.end());
        std::sort(ordered.begin(), ordered.end());
        if (used + ordered.size() > capacity) return SIZE_MAX;
        for (auto& v : ordered) {
            rows[used] = v.first;
            masks[used] = v.second;
            ++used;
        }
        offsets[i + 1] = used;
    }
    return used;
}

extern "C" size_t ber_merge(const uint32_t* ar, const uint16_t* am, const uint32_t* ao,
                           const uint32_t* br, const uint16_t* bm, const uint32_t* bo,
                           size_t n, uint32_t* rows, uint16_t* masks, uint32_t* offsets) {
    size_t used = 0;
    offsets[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t a = ao[i], b = bo[i];
        while (a < ao[i+1] || b < bo[i+1]) {
            if (b == bo[i+1] || (a < ao[i+1] && ar[a] < br[b])) {
                rows[used] = ar[a]; masks[used++] = am[a++];
            } else if (a == ao[i+1] || br[b] < ar[a]) {
                rows[used] = br[b]; masks[used++] = bm[b++];
            } else {
                rows[used] = ar[a]; masks[used++] = am[a++] | bm[b++];
            }
        }
        offsets[i+1] = used;
    }
    return used;
}
