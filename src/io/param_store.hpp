#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>
#include "core/hydro_config.hpp"

/*
 All "key value" pairs from the parameter file.

 Every part of the code (readParams, each problem generator) reads the keys it needs from here.
 The store remembers which keys were read, 
 so after setup the run can warn about keys that nobody used: 
 almost always a typo ("limitter mc" would otherwise silently run with the default limiter).
*/

class ParamStore{
  private:
    //in params: key + val
    static Real to_real(const std::string& key, const std::string& s){ //input Real vals
      try{
        size_t pos = 0;
        Real v = std::stod(s, &pos);
        if(pos == s.size()) return v;
      }catch(const std::exception&){}
      throw std::runtime_error("Parameter error: '" + key + 
                               "' must be a number, got '" + s + "'");
    }
    static int to_int(const std::string& key, const std::string& s){ //input int vals
      try{
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if(pos == s.size()) return v;
      }catch(const std::exception&){}
      throw std::runtime_error("Parameter error: '" + key + 
                               "' must be an integer, got '" + s + "'");
    }

    std::map<std::string, std::string> kv_;
    mutable std::set<std::string> used_; //reading a key is logically const, but it is recorded

  public:
    void set(const std::string& key, const std::string& value){ kv_[key] = value; }

    //true if the key is in the file (counts as "used": the caller looked at it on purpose)
    bool has(const std::string& key) const {
      bool found = kv_.count(key) > 0;
      if(found) used_.insert(key);
      return found;
    }

    //required keys: a missing key is an error
    std::string get(const std::string& key) const {
      auto it = kv_.find(key);
      if(it == kv_.end()) throw std::runtime_error("Parameter error: missing parameter '" + 
                                                    key + "'");
      used_.insert(key);
      return it->second;
    }

    Real get_real(const std::string& key) const { return to_real(key, get(key)); }
    int  get_int (const std::string& key) const { return to_int(key, get(key)); }

    //optional keys: the default is used if the key is not in the file
    std::string get(const std::string& key, const std::string& def) const {
      return has(key) ? get(key) : def;
    }

    Real get_real(const std::string& key, Real def) const {
      return has(key) ? get_real(key) : def;
    }

    int  get_int (const std::string& key, int def)  const {
      return has(key) ? get_int(key)  : def;
    }

    //keys in the file that nothing has read (so far)
    std::vector<std::string> unused() const {
      std::vector<std::string> out;
      for(const auto& kv : kv_) if(!used_.count(kv.first)) out.push_back(kv.first);
      return out;
    }



};
