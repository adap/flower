#!/bin/sh

# Defining some language parameters to add to the outdated config files
language_config=$(cat <<-END
locale_dirs = ['../locales']
gettext_compact = 'framework-docs'
END
)

# Defining the code necessary to add to the outdated config files for the version switcher
version_switcher=$(cat <<-END
html_context = dict()
html_context['current_language'] = '$current_language'
from git import Repo
repo = Repo( search_parent_directories=True )
current_version = '$current_version'
html_context['current_version'] = {}
html_context['current_version']['url'] = current_version
html_context['current_version']['full_name'] = 'main' if current_version=='main' else f'Flower Framework {current_version}'
html_context['versions'] = list()
versions = [tag.name for tag in repo.tags if int(tag.name[1]) != 0]
versions.append('main')
for version in versions:
    html_context['versions'].append({'name': version})
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
        'sidebar/versioning.html',
        'sidebar/lang.html',
    ]
}
END
)

# Store the current branch in a variable
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Move to the top level
cd $(git rev-parse --show-toplevel)

# Clean up previous builds
rm -rf doc/build

# Create a temporary directory and store locales and _templates files in it
tmp_dir=`mktemp -d`
cp -r doc/locales ${tmp_dir}/locales
cp -r doc/source/_templates ${tmp_dir}/_templates
cp doc/source/conf.py ${tmp_dir}/conf.py

cd doc

# Get a list of languages based on the folders in locales
languages="en `find locales/ -mindepth 1 -maxdepth 1 -type d -exec basename '{}' \;`"
# Get a list of tags, excluding those before v1.0.0
versions="`git for-each-ref '--format=%(refname:lstrip=-1)' refs/tags/ | grep -iE '^v(([1-9]|[0-9]{2,}).*)$'`"

for current_version in ${versions}; do
  
   # Make the current language available to conf.py
   export current_version
   git checkout ${current_version}
   echo "INFO: Building sites for ${current_version}"
  
   # Skip this branch if it doesn't have our docs dir & sphinx config
   if [ ! -e 'source/conf.py' ]; then
      echo "INFO: Couldn't find 'doc/source/conf.py' (skipped)"
      continue
   fi

   changed=false
  
   for current_language in ${languages}; do
  
      # Make the current language available to conf.py
      export current_language
  
      echo "INFO: Building for ${current_language}"

      if [ ! -f "locales/$current_language/LC_MESSAGES/framework-docs.po" ]; then

        # Adding translation to versions that didn't contain one
        if [ $current_language != "en" ]; then
          echo "No translation, using default one"

          # Remove any previous file in locales
          rm -rf locales/$current_language
          mkdir -p locales

          # Copy updated version of locales
          cp -r ${tmp_dir}/locales/$current_language locales/
          # Add necessary config to conf.py
          echo "$language_config" >> source/conf.py

          # Update the text and the translation to match the source files
          make gettext
          sphinx-intl update -p build/gettext -l ${current_language}
        fi

        # Copy updated version of html files
        cp -r ${tmp_dir}/_templates source
        # Adding version switcher to conf.py of versions that didn't contain it
        echo "$version_switcher" >> source/conf.py

        changed=true
      fi
  
      # Actually building the docs for a given language and version
      sphinx-build -b html source/ build/html/${current_version}/${current_language} -A lang=True -D language=${current_language}

      # Restore branch as it was to avoid conflicts
      if [ changed ]; then
        git restore source/conf.py
        rm -rf locales/${current_language}
        rm -rf source/_templates/sidebar
        git restore source/_templates/base.html
      fi
   done
done
  
# Build the main version (main for GH CI, local branch for local) 
if [ $GITHUB_ACTIONS ]
then
  git switch main
else
  git switch $current_branch
fi

current_version=main
export current_version
for current_language in ${languages}; do
    export current_language
    sphinx-build -b html source/ build/html/${current_version}/${current_language} -A lang=True -D language=${current_language}
done

# Copy main version to the root of the built docs
cp -r build/html/main/en/* build/html/
