#!/bin/sh

current_branch=$(git rev-parse --abbrev-ref HEAD)

cd $(git rev-parse --show-toplevel)

##############
# BUILD DOCS #
##############
  
# first, cleanup any old builds' static assets
# make -C doc clean
rm -rf doc/build/html
rm -rf doc/build/gettext

tmp_dir=`mktemp -d`

cp -r doc/locales ${tmp_dir}/locales
cp -r doc/source/_templates ${tmp_dir}/_templates
cp doc/source/conf.py ${tmp_dir}/conf.py

cd doc

# get a list of branches, excluding 'HEAD' and 'gh-pages'
languages="en `find locales/ -mindepth 1 -maxdepth 1 -type d -exec basename '{}' \;`"
versions="`git for-each-ref '--format=%(refname:lstrip=-1)' refs/tags/ | grep -iE '^v(([1-9]|[0-9]{2,}).*)$'`"
for current_version in ${versions}; do
  
   # make the current language available to conf.py
   export current_version

   changed=false

   git checkout ${current_version}
  
   echo "INFO: Building sites for ${current_version}"
  
   # skip this branch if it doesn't have our docs dir & sphinx config
   if [ ! -e 'source/conf.py' ]; then
      echo "INFO: Couldn't find 'doc/source/conf.py' (skipped)"
      continue
   fi
  
   for current_language in ${languages}; do
  
      # make the current language available to conf.py
      export current_language
  
      echo "INFO: Building for ${current_language}"

      if [[ ! -f "locales/$current_language/LC_MESSAGES/framework-docs.po" ]]; then
        if [[ $current_language != "en" ]]; then
          echo "No translation, using default one"
          rm -rf locales/$current_language
          mkdir -p locales
          cp -r ${tmp_dir}/locales/$current_language locales/
          echo "locale_dirs = ['../locales']" >> source/conf.py
          echo "gettext_compact = 'framework-docs'" >> source/conf.py
          make gettext
          sphinx-intl update -p build/gettext -l ${current_language}
        fi
        cp -r ${tmp_dir}/_templates source

        echo "html_context = dict()" >> source/conf.py
        echo "html_context['current_language'] = '$current_language'" >> source/conf.py
        echo "from git import Repo" >> source/conf.py
        echo "repo = Repo( search_parent_directories=True )" >> source/conf.py
        echo "current_version = '$current_version'" >> source/conf.py
        echo "html_context['current_version'] = {}" >> source/conf.py
        echo "html_context['current_version']['name'] = 'Main' if current_version=='main' else current_version" >> source/conf.py
        echo "html_context['current_version']['url'] = current_version" >> source/conf.py
        echo "html_context['current_version']['full_name'] = 'Main' if current_version=='main' else f'Flower Framework {current_version}'" >> source/conf.py
        echo "html_context['versions'] = list()" >> source/conf.py
        echo "versions = [tag.name for tag in repo.tags if int(tag.name[1]) != 0]" >> source/conf.py
        echo "versions.append('main')" >> source/conf.py
        echo "for version in versions:" >> source/conf.py
        echo "    html_context['versions'].append({'name': version})" >> source/conf.py
        echo "html_sidebars = {" >> source/conf.py
        echo "    '**': [" >> source/conf.py
        echo "        'sidebar/brand.html'," >> source/conf.py
        echo "        'sidebar/search.html'," >> source/conf.py
        echo "        'sidebar/scroll-start.html'," >> source/conf.py
        echo "        'sidebar/navigation.html'," >> source/conf.py
        echo "        'sidebar/scroll-end.html'," >> source/conf.py
        echo "        'sidebar/versioning.html'," >> source/conf.py
        echo "        'sidebar/lang.html'," >> source/conf.py
        echo "    ]" >> source/conf.py
        echo "}" >> source/conf.py

        changed=true
      fi
  
      sphinx-build -b html source/ build/html/${current_version}/${current_language} -A lang=True -D language=${current_language}

      if [[ changed ]]; then
        git restore source/conf.py
        rm -rf locales/${current_language}
        rm -rf source/_templates/sidebar
        git restore source/_templates/base.html
      fi
  
   done
  
done
  
git switch $current_branch
current_version=main
export current_version
for current_language in ${languages}; do
    export current_language
    sphinx-build -b html source/ build/html/${current_version}/${current_language} -A lang=True -D language=${current_language}
done

cp -r build/html/main/en/* build/html/
