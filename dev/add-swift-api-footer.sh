#!/bin/bash

# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

footer_template_file="dev/swift-docs-resources/footer.html"
footer_template=$(<"$footer_template_file")
placeholder='<body data-color-scheme="auto">'
html_file="Swiftdoc/html/documentation/flwr/index.html"

sed -i '' -e "s#$(printf '%s' "$placeholder" | sed 's/[&/\]/\\&/g')#$placeholder\n$footer_template#" "$html_file"

flower_logo='dev/swift-docs-resources/logo_secondary-w-border.png'
footer_template_css="dev/swift-docs-resources/footer.css"
destination_images="Swiftdoc/html/images/"
destination_css="Swiftdoc/html/css/"

cp "$flower_logo" "$destination_images"
cp "$footer_template_css" "$destination_css"
