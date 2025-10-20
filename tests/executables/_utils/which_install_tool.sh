#!/usr/bin/env bash

command_exists() {
	command -v "$@" > /dev/null 2>&1
}

# install template
# determine whether the user is root mode to execute this script
# prefix_sudo=""
# current_user=$(whoami)
# if [ "$current_user" != "root" ]; then
# 	echo "User $current_user need to add sudo permission keywords"
#	prefix_sudo="sudo"
# fi
#
# echo "prefix_sudo= $prefix_sudo"
#
# if command_exists apt; then
# 	$prefix_sudo apt install -y 
# elif command_exists dnf; then
# 	$prefix_sudo dnf install -y 
# else
# 	$prefix_sudo yum install -y 
# fi
