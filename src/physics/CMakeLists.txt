#src/physics: physics modules (disk model, later conduction, cooling, ...)
file(GLOB PHYSICS_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(szuchydro_physics STATIC ${PHYSICS_SOURCES})
target_link_libraries(szuchydro_physics PUBLIC szuchydro_mesh)
