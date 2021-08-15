GRPC libraries are generated on Windows, so the relevant tools are "exe" files(protoc.exe, grpc_cpp_plugin.exe).
"Dependent libary" should be used for c++ client building.
zlib.dll should be in the same directory as the running client, otherwise it will get error (when build by Visual Studio)
