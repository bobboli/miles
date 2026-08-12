#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>

typedef int (*shm_unlink_fn)(const char *name);

int shm_unlink(const char *name) {
    static shm_unlink_fn next_shm_unlink;

    if (next_shm_unlink == NULL) {
        next_shm_unlink = (shm_unlink_fn)dlsym(RTLD_NEXT, "shm_unlink");
        if (next_shm_unlink == NULL) {
            errno = ENOSYS;
            return -1;
        }
    }

    int result = next_shm_unlink(name);
    if (result == -1 && errno == ENOENT) {
        errno = 0;
        return 0;
    }
    return result;
}
