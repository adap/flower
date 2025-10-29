# #!/bin/bash
# # Compare conf.py across different versions of Flower
# # Usage:
# #   ./cmp.sh v1.8.0 v1.9.0        # Compare two specific versions
# #   ./cmp.sh                       # Compare all consecutive versions

# set -e

# # Move to the repository root
# cd "$(git rev-parse --show-toplevel)"

# # Function to determine the correct path for conf.py based on version
# get_conf_path() {
#     local version=$1
    
#     # Extract version number for comparison (remove 'v' prefix and convert to number)
#     # v1.14.0 and later use framework/docs/source/conf.py
#     # Earlier versions use doc/source/conf.py
    
#     # Simple version comparison: if >= v1.14.0, use new path
#     if [[ "$version" == "main" ]]; then
#         echo "framework/docs/source/conf.py"
#     else
#         # Extract major and minor version (e.g., v1.14.0 -> 1.14)
#         local ver_num=$(echo "$version" | sed 's/^v//' | cut -d. -f1,2)
#         local major=$(echo "$ver_num" | cut -d. -f1)
#         local minor=$(echo "$ver_num" | cut -d. -f2)
        
#         # Check if version >= 1.14
#         if (( major > 1 )) || (( major == 1 && minor >= 14 )); then
#             echo "framework/docs/source/conf.py"
#         else
#             echo "doc/source/conf.py"
#         fi
#     fi
# }

# # Function to compare two versions
# compare_versions() {
#     local v1=$1
#     local v2=$2
    
#     local path1=$(get_conf_path "$v1")
#     local path2=$(get_conf_path "$v2")
    
#     echo "========================================"
#     echo "Comparing: $v1 → $v2"
#     echo "  $v1: $path1"
#     echo "  $v2: $path2"
#     echo "========================================"
    
#     # Check if files exist
#     if ! git cat-file -e "$v1:$path1" 2>/dev/null; then
#         echo "ERROR: File not found at $v1:$path1"
#         return 1
#     fi
    
#     if ! git cat-file -e "$v2:$path2" 2>/dev/null; then
#         echo "ERROR: File not found at $v2:$path2"
#         return 1
#     fi
    
#     # Show the diff
#     git diff --color "$v1:$path1" "$v2:$path2"
#     echo ""
# }

# # Main logic
# if [ $# -eq 2 ]; then
#     # Two versions provided - compare them directly
#     compare_versions "$1" "$2"
# elif [ $# -eq 0 ]; then
#     # No versions provided - compare all consecutive versions
#     echo "Fetching all version tags (v1.8.0 and later)..."
    
#     # Get sorted list of version tags >= v1.8.0
#     versions=$(git for-each-ref '--format=%(refname:lstrip=-1)' refs/tags/ | \
#         grep -iE '^v((([1-9]|[0-9]{2,}).*\.([8-9]|[0-9]{2,}).*)|([2-9]|[0-9]{2,}).*)$' | \
#         sort -V)
    
#     # Convert to array
#     version_array=($versions)
    
#     # Add main as the last version
#     version_array+=("main")
    
#     echo "Found ${#version_array[@]} versions to compare"
#     echo ""
    
#     # Compare consecutive versions
#     for ((i=0; i<${#version_array[@]}-1; i++)); do
#         v1="${version_array[$i]}"
#         v2="${version_array[$((i+1))]}"
        
#         compare_versions "$v1" "$v2"
        
#         # Pause after each comparison (optional)
#         # read -p "Press Enter to continue to next comparison..."
#     done
    
#     echo "========================================"
#     echo "Comparison complete!"
#     echo "========================================"
# else
#     echo "Usage:"
#     echo "  $0 <version1> <version2>    # Compare two specific versions"
#     echo "  $0                           # Compare all consecutive versions"
#     echo ""
#     echo "Examples:"
#     echo "  $0 v1.8.0 v1.9.0"
#     echo "  $0 v1.13.0 main"
#     echo "  $0"
#     exit 1
# fi
set -e

# Get all version tags >= v1.8.0, sorted
versions=$(git for-each-ref '--format=%(refname:lstrip=-1)' refs/tags/ | \
    grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | \
    sort -V | \
    awk -F'[v.]' '{ if ($2 > 1 || ($2 == 1 && $3 >= 8)) print $0 }')
# versions="`git for-each-ref '--format=%(refname:lstrip=-1)' refs/tags/ | grep -iE '^v((([1-9]|[0-9]{2,}).*\.([8-9]|[0-9]{2,}).*)|([2-9]|[0-9]{2,}).*)$'`"

for current_version in ${versions}; do

    echo "Processing version: ${current_version}"

done