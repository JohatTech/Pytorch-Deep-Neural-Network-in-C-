# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/johattech/Pytorch_projects

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/johattech/Pytorch_projects/build

# Include any dependencies generated for this target.
include CMakeFiles/pytorch_project.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pytorch_project.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pytorch_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pytorch_project.dir/flags.make

CMakeFiles/pytorch_project.dir/main.cpp.o: CMakeFiles/pytorch_project.dir/flags.make
CMakeFiles/pytorch_project.dir/main.cpp.o: ../main.cpp
CMakeFiles/pytorch_project.dir/main.cpp.o: CMakeFiles/pytorch_project.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/johattech/Pytorch_projects/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pytorch_project.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pytorch_project.dir/main.cpp.o -MF CMakeFiles/pytorch_project.dir/main.cpp.o.d -o CMakeFiles/pytorch_project.dir/main.cpp.o -c /home/johattech/Pytorch_projects/main.cpp

CMakeFiles/pytorch_project.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pytorch_project.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/johattech/Pytorch_projects/main.cpp > CMakeFiles/pytorch_project.dir/main.cpp.i

CMakeFiles/pytorch_project.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pytorch_project.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/johattech/Pytorch_projects/main.cpp -o CMakeFiles/pytorch_project.dir/main.cpp.s

# Object files for target pytorch_project
pytorch_project_OBJECTS = \
"CMakeFiles/pytorch_project.dir/main.cpp.o"

# External object files for target pytorch_project
pytorch_project_EXTERNAL_OBJECTS =

pytorch_project: CMakeFiles/pytorch_project.dir/main.cpp.o
pytorch_project: CMakeFiles/pytorch_project.dir/build.make
pytorch_project: /home/johattech/libtorch/lib/libtorch.so
pytorch_project: /home/johattech/libtorch/lib/libc10.so
pytorch_project: /home/johattech/libtorch/lib/libkineto.a
pytorch_project: /home/johattech/libtorch/lib/libc10.so
pytorch_project: CMakeFiles/pytorch_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/johattech/Pytorch_projects/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pytorch_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pytorch_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pytorch_project.dir/build: pytorch_project
.PHONY : CMakeFiles/pytorch_project.dir/build

CMakeFiles/pytorch_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pytorch_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pytorch_project.dir/clean

CMakeFiles/pytorch_project.dir/depend:
	cd /home/johattech/Pytorch_projects/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/johattech/Pytorch_projects /home/johattech/Pytorch_projects /home/johattech/Pytorch_projects/build /home/johattech/Pytorch_projects/build /home/johattech/Pytorch_projects/build/CMakeFiles/pytorch_project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pytorch_project.dir/depend

