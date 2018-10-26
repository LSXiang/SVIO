# Add Sources
set(FATS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/fast_10.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/fast_10_score.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/nonmax_3x3.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/faster_corner_10_sse.cpp)

# Add library
INCLUDE_DIRECTORIES(include src)
ADD_LIBRARY(fast_LIBRARIES SHARED ${FATS_SOURCES})


