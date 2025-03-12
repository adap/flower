/* eslint-disable @typescript-eslint/no-unnecessary-condition */
export const isNode: boolean =
  typeof process !== 'undefined' && process.versions?.node != null;
/* eslint-enable @typescript-eslint/no-unnecessary-condition */
