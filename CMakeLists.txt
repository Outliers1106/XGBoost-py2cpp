CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)


PROJECT(xgb-infer)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

add_executable(xgb-infer xgb-infer.cpp)
INCLUDE_DIRECTORIES("./xgb-include/" "./eigen-3.3.7/" "./")
LINK_DIRECTORIES("./xgb-lib")
find_library(XGBOOST xgboost HINTS "./xgb-lib")
target_link_libraries(xgb-infer PUBLIC ${XGBOOST})

