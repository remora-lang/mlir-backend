CMAKE_MINIMUM_REQUIRED(VERSION 3.7 FATAL_ERROR)
find_package(antlr4-runtime CONFIG REQUIRED)
find_package(antlr4-generator CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(ANTLR4_JAR  ${CMAKE_SOURCE_DIR}/third-party/antlr/antlr-4.13.2-complete.jar)
set(GRAMMAR_FILE  ${CMAKE_SOURCE_DIR}/Futhark.g4)
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_custom_command(
    OUTPUT "${GEN_DIR}/FutharkLexer.cpp" "${GEN_DIR}/FutharkParser.cpp"
    COMMAND java -jar ${ANTLR4_JAR} -Dlanguage=Cpp -o ${GEN_DIR} ${GRAMMAR_FILE}
    DEPENDS ${GRAMMAR_FILE}
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
        )

add_executable  (mlir-backend "mlir-backend.cpp" "${GEN_DIR}/FutharkLexer.cpp" "${GEN_DIR}/FutharkParser.cpp" "ir/core.hpp" "ir/parser.hpp" "ir/primitive.hpp" "ir/soac.hpp" "ir/syntax.hpp" "ir/compiler.hpp")

target_precompile_headers(mlir-backend PRIVATE 
    pch.h
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

target_include_directories(mlir-backend PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "Build type of mlir-backend is : ${CMAKE_BUILD_TYPE}")
target_link_libraries(mlir-backend PRIVATE ${LIBS})
target_compile_options(mlir-backend PRIVATE -frtti)

target_include_directories(mlir-backend PRIVATE "${ANTLR4_INCLUDE_DIR}")
target_include_directories(mlir-backend PRIVATE ${GEN_DIR})

if (WIN32)
	target_link_libraries(mlir-backend PRIVATE antlr4_shared)
else()
	target_link_libraries(mlir-backend PRIVATE antlr4_static)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET mlir-backend PROPERTY CXX_STANDARD 20)
endif()
