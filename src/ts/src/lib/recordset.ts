export type ConfigsRecordValue =
  | string
  | bigint
  | number
  | boolean
  | (string | bigint | number | boolean)[];
export type MetricsRecordValue = number | bigint | (number | bigint)[];

export class ArrayData {
  constructor(
    public dtype: string,
    public shape: number[],
    public stype: string,
    public data: Uint8Array,
  ) {}
}

export class ParametersRecord {
  [key: string]: ArrayData;

  constructor(data: { [key: string]: ArrayData } = {}) {
    Object.assign(this, data);
  }
}

export class MetricsRecord {
  [key: string]: MetricsRecordValue;

  constructor(data: { [key: string]: MetricsRecordValue } = {}) {
    Object.assign(this, data);
  }
}

export class ConfigsRecord {
  [key: string]: ConfigsRecordValue;

  constructor(data: { [key: string]: ConfigsRecordValue } = {}) {
    Object.assign(this, data);
  }
}

export class RecordSet {
  public parametersRecords: { [key: string]: ParametersRecord } = {};
  public metricsRecords: { [key: string]: MetricsRecord } = {};
  public configsRecords: { [key: string]: ConfigsRecord } = {};

  constructor(
    parametersRecords: { [key: string]: ParametersRecord } = {},
    metricsRecords: { [key: string]: MetricsRecord } = {},
    configsRecords: { [key: string]: ConfigsRecord } = {},
  ) {
    this.parametersRecords = parametersRecords;
    this.metricsRecords = metricsRecords;
    this.configsRecords = configsRecords;
  }
}
