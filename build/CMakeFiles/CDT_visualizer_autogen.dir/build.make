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

# Utility rule file for CDT_visualizer_autogen.

# Include the progress variables for this target.
include CMakeFiles/CDT_visualizer_autogen.dir/progress.make

CMakeFiles/CDT_visualizer_autogen:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/luziang/桌面/PointTexture/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic MOC for target CDT_visualizer"
	/usr/bin/cmake -E cmake_autogen /home/luziang/桌面/PointTexture/build/CMakeFiles/CDT_visualizer_autogen.dir/AutogenInfo.json Debug

CDT_visualizer_autogen: CMakeFiles/CDT_visualizer_autogen
CDT_visualizer_autogen: CMakeFiles/CDT_visualizer_autogen.dir/build.make

.PHONY : CDT_visualizer_autogen

# Rule to build all files generated by this target.
CMakeFiles/CDT_visualizer_autogen.dir/build: CDT_visualizer_autogen

.PHONY : CMakeFiles/CDT_visualizer_autogen.dir/build

CMakeFiles/CDT_visualizer_autogen.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CDT_visualizer_autogen.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CDT_visualizer_autogen.dir/clean

CMakeFiles/CDT_visualizer_autogen.dir/depend:
	cd /home/luziang/桌面/PointTexture/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luziang/桌面/PointTexture/CDT/visualizer /home/luziang/桌面/PointTexture/CDT/visualizer /home/luziang/桌面/PointTexture/build /home/luziang/桌面/PointTexture/build /home/luziang/桌面/PointTexture/build/CMakeFiles/CDT_visualizer_autogen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CDT_visualizer_autogen.dir/depend

