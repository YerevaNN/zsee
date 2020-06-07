local util = import 'util.libsonnet';

local enable_cache = false;
local auto_cache = true;
local salt = 'aX&hack_';
local cache_dir = 'caches';

{
    hash(config)::
        std.md5(salt + util.stringify(config)),

    get_dir(config)::
        if enable_cache
            then cache_dir + '/' + self.hash(config)
            else null,

    get_update(config, enabled=null):: 
        local cache_dir = self.get_dir(config);
        { cache_directory: cache_dir },

    resolve_enabled(enabled=null)::
        if enabled == null
            then auto_cache
            else enabled,

    apply(config, enabled=true)::
        config + if self.resolve_enabled(enabled)
                    then self.get_update(config)
                    else {}
}