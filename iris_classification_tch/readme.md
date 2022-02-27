- to link libtorch.so, find it via `find /path/to/file/ -iname filename`, then use these 
```
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

## reference 

- https://github.com/LaurentMazare/tch-rs/tree/main/examples/mnist