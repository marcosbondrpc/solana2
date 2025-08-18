cmake_minimum_required(VERSION 3.18)
project(treelite_abi_wrapper LANGUAGES CXX)

# C++ standard and optimization settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Input parameters from presets
set(DEFAULT_MEV_INNER_LIB "" CACHE STRING "Default inner MEV treelite .so path")
set(DEFAULT_MEV_CALIB "" CACHE STRING "Default MEV calibrator json path")
set(DEFAULT_ARB_INNER_LIB "" CACHE STRING "Default inner ARB treelite .so path")
set(DEFAULT_ARB_CALIB "" CACHE STRING "Default ARB calibrator json path")
set(GIT_SHA "dev" CACHE STRING "Git short sha for SONAME suffix")

# Find required libraries
find_library(TREELITE_RUNTIME NAMES treelite_runtime REQUIRED
    HINTS /usr/local/lib /usr/lib ${CMAKE_PREFIX_PATH}/lib)

# Common compiler flags for maximum performance
set(COMMON_FLAGS
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
    -ftree-vectorize
    -fomit-frame-pointer
    -flto
    -fno-plt
    -fvisibility=hidden
    -fno-semantic-interposition
    -mfma
    -mavx2
    -mbmi2
    -DNDEBUG
    -pipe
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Link-time optimization flags
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# MEV Model Library
add_library(mev_model SHARED treelite_abi_wrapper.cc)
target_compile_definitions(mev_model PRIVATE
    DEFAULT_INNER_LIB_PATH="${DEFAULT_MEV_INNER_LIB}"
    DEFAULT_CALIB_PATH="${DEFAULT_MEV_CALIB}"
    MODEL_TYPE="MEV"
)
target_compile_options(mev_model PRIVATE ${COMMON_FLAGS})
target_link_libraries(mev_model PRIVATE 
    ${TREELITE_RUNTIME} 
    dl 
    pthread
    m
    atomic
)
target_link_options(mev_model PRIVATE
    -Wl,-O3
    -Wl,--gc-sections
    -Wl,--as-needed
    -Wl,-z,now
    -Wl,-z,relro
    -Wl,--exclude-libs,ALL
)
set_target_properties(mev_model PROPERTIES 
    OUTPUT_NAME "mev_v${GIT_SHA}"
    SOVERSION 1
    VERSION 1.0.0
)

# ARB Model Library
add_library(arb_model SHARED treelite_abi_wrapper.cc)
target_compile_definitions(arb_model PRIVATE
    DEFAULT_INNER_LIB_PATH="${DEFAULT_ARB_INNER_LIB}"
    DEFAULT_CALIB_PATH="${DEFAULT_ARB_CALIB}"
    MODEL_TYPE="ARB"
)
target_compile_options(arb_model PRIVATE ${COMMON_FLAGS})
target_link_libraries(arb_model PRIVATE 
    ${TREELITE_RUNTIME} 
    dl 
    pthread
    m
    atomic
)
target_link_options(arb_model PRIVATE
    -Wl,-O3
    -Wl,--gc-sections
    -Wl,--as-needed
    -Wl,-z,now
    -Wl,-z,relro
    -Wl,--exclude-libs,ALL
)
set_target_properties(arb_model PROPERTIES 
    OUTPUT_NAME "arb_v${GIT_SHA}"
    SOVERSION 1
    VERSION 1.0.0
)

# Installation rules
install(TARGETS mev_model arb_model
    LIBRARY DESTINATION /home/kidgordones/0solana/node/rust-services/shared/treelite
    RUNTIME DESTINATION /home/kidgordones/0solana/node/rust-services/shared/treelite
)

# Post-build strip for smaller binaries
add_custom_command(TARGET mev_model POST_BUILD
    COMMAND ${CMAKE_STRIP} -s $<TARGET_FILE:mev_model>
    COMMENT "Stripping MEV model library"
)

add_custom_command(TARGET arb_model POST_BUILD
    COMMAND ${CMAKE_STRIP} -s $<TARGET_FILE:arb_model>
    COMMENT "Stripping ARB model library"
)

# Generate metadata files
add_custom_command(TARGET mev_model POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
    "{\"model\":\"MEV\",\"version\":\"${GIT_SHA}\",\"lib\":\"$<TARGET_FILE_NAME:mev_model>\"}"
    > $<TARGET_FILE_DIR:mev_model>/mev_metadata.json
    COMMENT "Generating MEV metadata"
)

add_custom_command(TARGET arb_model POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
    "{\"model\":\"ARB\",\"version\":\"${GIT_SHA}\",\"lib\":\"$<TARGET_FILE_NAME:arb_model>\"}"
    > $<TARGET_FILE_DIR:arb_model>/arb_metadata.json
    COMMENT "Generating ARB metadata"
)

# Print configuration summary
message(STATUS "=====================================")
message(STATUS "Treelite ABI Wrapper Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Git SHA: ${GIT_SHA}")
message(STATUS "  MEV Inner Lib: ${DEFAULT_MEV_INNER_LIB}")
message(STATUS "  ARB Inner Lib: ${DEFAULT_ARB_INNER_LIB}")
message(STATUS "  Treelite Runtime: ${TREELITE_RUNTIME}")
message(STATUS "=====================================")