# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luziang/桌面/PointTexture/CDT/visualizer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luziang/桌面/PointTexture/build

# Include any dependencies generated for this target.
include CMakeFiles/CDT_visualizer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CDT_visualizer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CDT_visualizer.dir/flags.make

CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.o: CMakeFiles/CDT_visualizer.dir/flags.make
CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.o: CDT_visualizer_autogen/mocs_compilation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luziang/桌面/PointTexture/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.o"
	/usr/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.o -c /home/luziang/桌面/PointTexture/build/CDT_visualizer_autogen/mocs_compilation.cpp

CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.i"
	/usr/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luziang/桌面/PointTexture/build/CDT_visualizer_autogen/mocs_compilation.cpp > CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.i

CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.s"
	/usr/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luziang/桌面/PointTexture/build/CDT_visualizer_autogen/mocs_compilation.cpp -o CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.s

CMakeFiles/CDT_visualizer.dir/main.cpp.o: CMakeFiles/CDT_visualizer.dir/flags.make
CMakeFiles/CDT_visualizer.dir/main.cpp.o: /home/luziang/桌面/PointTexture/CDT/visualizer/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luziang/桌面/PointTexture/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CDT_visualizer.dir/main.cpp.o"
	/usr/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CDT_visualizer.dir/main.cpp.o -c /home/luziang/桌面/PointTexture/CDT/visualizer/main.cpp

CMakeFiles/CDT_visualizer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CDT_visualizer.dir/main.cpp.i"
	/usr/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luziang/桌面/PointTexture/CDT/visualizer/main.cpp > CMakeFiles/CDT_visualizer.dir/main.cpp.i

CMakeFiles/CDT_visualizer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CDT_visualizer.dir/main.cpp.s"
	/usr/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luziang/桌面/PointTexture/CDT/visualizer/main.cpp -o CMakeFiles/CDT_visualizer.dir/main.cpp.s

# Object files for target CDT_visualizer
CDT_visualizer_OBJECTS = \
"CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.o" \
"CMakeFiles/CDT_visualizer.dir/main.cpp.o"

# External object files for target CDT_visualizer
CDT_visualizer_EXTERNAL_OBJECTS =

CDT_visualizer: CMakeFiles/CDT_visualizer.dir/CDT_visualizer_autogen/mocs_compilation.cpp.o
CDT_visualizer: CMakeFiles/CDT_visualizer.dir/main.cpp.o
CDT_visualizer: CMakeFiles/CDT_visualizer.dir/build.make
CDT_visualizer: /home/luziang/anaconda3/lib/libQt5Widgets.so.5.15.2
CDT_visualizer: /home/luziang/anaconda3/lib/libQt5Gui.so.5.15.2
CDT_visualizer: /home/luziang/anaconda3/lib/libQt5Core.so.5.15.2
CDT_visualizer: CMakeFiles/CDT_visualizer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luziang/桌面/PointTexture/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CDT_visualizer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CDT_visualizer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CDT_visualizer.dir/build: CDT_visualizer

.PHONY : CMakeFiles/CDT_visualizer.dir/build

CMakeFiles/CDT_visualizer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CDT_visualizer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CDT_visualizer.dir/clean

CMakeFiles/CDT_visualizer.dir/depend:
	cd /home/luziang/桌面/PointTexture/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luziang/桌面/PointTexture/CDT/visualizer /home/luziang/桌面/PointTexture/CDT/visualizer /home/luziang/桌面/PointTexture/build /home/luziang/桌面/PointTexture/build /home/luziang/桌面/PointTexture/build/CMakeFiles/CDT_visualizer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CDT_visualizer.dir/depend

