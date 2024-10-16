"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.warnPreviewFeature = warnPreviewFeature;
exports.warnDeprecatedFeature = warnDeprecatedFeature;
exports.warnDeprecatedFeatureWithExample = warnDeprecatedFeatureWithExample;
exports.warnUnsupportedFeature = warnUnsupportedFeature;
function warnPreviewFeature(name) {
    console.warn(`PREVIEW FEATURE: ${name}

            This is a preview feature.It could change significantly or be removed
            entirely in future versions of Flower.
        `);
}
function warnDeprecatedFeature(name) {
    console.warn(`DEPRECATED FEATURE: ${name}

            This is a deprecated feature.It will be removed
            entirely in future versions of Flower.
        `);
}
function warnDeprecatedFeatureWithExample(deprecation_message, example_message, code_example) {
    console.warn(`DEPRECATED FEATURE: ${deprecation_message}

            Check the following \`FEATURE UPDATE\` warning message for the preferred
            new mechanism to use this feature in Flower.
        `);
    console.warn(`FEATURE UPDATE: ${example_message}
        ------------------------------------------------------------
        ${code_example}
        ------------------------------------------------------------
       `);
}
function warnUnsupportedFeature(name) {
    console.warn(`UNSUPPORTED FEATURE: ${name}

            This is an unsupported feature.It will be removed
            entirely in future versions of Flower.
        `);
}
