# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

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
CMAKE_COMMAND = /home/alex/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/173.4548.31/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/alex/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/173.4548.31/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alex/Extra/GitHub/NeuralNetworksLearning/01

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/01.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/01.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/01.dir/flags.make

CMakeFiles/01.dir/train.cpp.o: CMakeFiles/01.dir/flags.make
CMakeFiles/01.dir/train.cpp.o: ../train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/01.dir/train.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/01.dir/train.cpp.o -c /home/alex/Extra/GitHub/NeuralNetworksLearning/01/train.cpp

CMakeFiles/01.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/01.dir/train.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alex/Extra/GitHub/NeuralNetworksLearning/01/train.cpp > CMakeFiles/01.dir/train.cpp.i

CMakeFiles/01.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/01.dir/train.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alex/Extra/GitHub/NeuralNetworksLearning/01/train.cpp -o CMakeFiles/01.dir/train.cpp.s

CMakeFiles/01.dir/train.cpp.o.requires:

.PHONY : CMakeFiles/01.dir/train.cpp.o.requires

CMakeFiles/01.dir/train.cpp.o.provides: CMakeFiles/01.dir/train.cpp.o.requires
	$(MAKE) -f CMakeFiles/01.dir/build.make CMakeFiles/01.dir/train.cpp.o.provides.build
.PHONY : CMakeFiles/01.dir/train.cpp.o.provides

CMakeFiles/01.dir/train.cpp.o.provides.build: CMakeFiles/01.dir/train.cpp.o


# Object files for target 01
01_OBJECTS = \
"CMakeFiles/01.dir/train.cpp.o"

# External object files for target 01
01_EXTERNAL_OBJECTS =

01: CMakeFiles/01.dir/train.cpp.o
01: CMakeFiles/01.dir/build.make
01: CMakeFiles/01.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 01"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/01.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/01.dir/build: 01

.PHONY : CMakeFiles/01.dir/build

CMakeFiles/01.dir/requires: CMakeFiles/01.dir/train.cpp.o.requires

.PHONY : CMakeFiles/01.dir/requires

CMakeFiles/01.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/01.dir/cmake_clean.cmake
.PHONY : CMakeFiles/01.dir/clean

CMakeFiles/01.dir/depend:
	cd /home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alex/Extra/GitHub/NeuralNetworksLearning/01 /home/alex/Extra/GitHub/NeuralNetworksLearning/01 /home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug /home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug /home/alex/Extra/GitHub/NeuralNetworksLearning/01/cmake-build-debug/CMakeFiles/01.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/01.dir/depend
