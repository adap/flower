export function warnPreviewFeature(name: string) {
  console.warn(
    `PREVIEW FEATURE: ${name}

            This is a preview feature.It could change significantly or be removed
            entirely in future versions of Flower.
        `,
  )
}

export function warnDeprecatedFeature(name: string) {
  console.warn(
    `DEPRECATED FEATURE: ${name}

            This is a deprecated feature.It will be removed
            entirely in future versions of Flower.
        `,
  )
}

export function warnDeprecatedFeatureWithExample(deprecation_message: string, example_message: string, code_example: string) {
  console.warn(
    `DEPRECATED FEATURE: ${deprecation_message}

            Check the following \`FEATURE UPDATE\` warning message for the preferred
            new mechanism to use this feature in Flower.
        `,
  )
  console.warn(
    `FEATURE UPDATE: ${example_message}
        ------------------------------------------------------------
        ${code_example}
        ------------------------------------------------------------
       `,
  )
}

export function warnUnsupportedFeature(name: string) {
  console.warn(
    `UNSUPPORTED FEATURE: ${name}

            This is an unsupported feature.It will be removed
            entirely in future versions of Flower.
        `,
  )
}
