// Copyright 2024 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
import toml from '@iarna/toml';
import { Server as GrpcServer } from '@grpc/grpc-js';
import { ChildProcess } from 'child_process';
import process from 'process';
import { readFileSync } from 'fs';
import { UserConfig, UserConfigValue } from './typing';

type OptionalArray<T> = T[] | null | undefined;

export function registerExitHandlers(
  grpcServers: OptionalArray<GrpcServer> = null,
  backgroundProcesses: OptionalArray<ChildProcess> = null
): void {
  // Store the original signal handlers so we can restore them later
  const defaultHandlers: { [key: string]: NodeJS.Process | null } = {
    SIGINT: null,
    SIGTERM: null,
  };

  const gracefulExitHandler = (signal: NodeJS.Signals) => {
    console.log(`Received ${signal}. Initiating graceful shutdown...`);

    // Restore default signal handlers
    process.removeListener(signal, gracefulExitHandler);

    // Stop the gRPC servers gracefully
    if (grpcServers) {
      grpcServers.forEach((grpcServer) => {
        grpcServer.tryShutdown((err) => {
          if (err) {
            console.error(`Error shutting down gRPC server: ${err}`);
          }
        });
      });
    }

    // Gracefully terminate background processes (if any)
    if (backgroundProcesses) {
      backgroundProcesses.forEach((proc) => {
        if (proc && proc.kill) {
          proc.kill('SIGTERM');
        }
      });
    }

    // Exit the process
    process.exit(0);
  };

  // Register the graceful exit handler for SIGINT and SIGTERM
  defaultHandlers.SIGINT = process.on('SIGINT', gracefulExitHandler);
  defaultHandlers.SIGTERM = process.on('SIGTERM', gracefulExitHandler);
}

export function flattenDict(
  rawDict: UserConfig | null | undefined,
  parentKey: string = ''
): UserConfig {
  if (!rawDict) {
    return {};
  }

  const items: [string, UserConfigValue][] = [];
  const separator = '.';

  for (const [key, value] of Object.entries(rawDict)) {
    const newKey = parentKey ? `${parentKey}${separator}${key}` : key;
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      items.push(...Object.entries(flattenDict(value as UserConfig, newKey)));
    } else if (isUserConfigValue(value)) {
      items.push([newKey, value]);
    } else {
      throw new Error(
        `The value for key ${key} needs to be of type 'int', 'float', 'bool', 'string', or a nested dictionary.`
      );
    }
  }

  return Object.fromEntries(items);
}

function isUserConfigValue(value: unknown): value is UserConfigValue {
  return (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean' ||
    (typeof value === 'object' && value !== null && !Array.isArray(value))
  );
}

export function parseConfigArgs(config: string[] | null | undefined): UserConfig {
  const overrides: UserConfig = {};

  if (!config || config.length === 0) {
    return overrides;
  }

  // Check if the input is a TOML file
  if (config.length === 1 && config[0].endsWith('.toml')) {
    const tomlPath = config[0];
    const fileContent = readFileSync(tomlPath, 'utf-8');
    const tomlData = loadToml(fileContent);
    return flattenDict(tomlData);
  }

  // Regular expression to capture key-value pairs with possible quoted values
  const pattern = /(\S+?)=('([^']*)'|"([^"]*)"|\S+)/g;

  for (const configLine of config) {
    if (configLine) {
      // Check if TOML files are mixed with key-value pairs
      if (configLine.endsWith('.toml')) {
        throw new Error('TOML files cannot be passed alongside key-value pairs.');
      }

      const matches = Array.from(configLine.matchAll(pattern));
      for (const [, key, , singleQuoteValue, doubleQuoteValue, unquotedValue] of matches) {
        const value = singleQuoteValue || doubleQuoteValue || unquotedValue;
        overrides[key] = parseValue(value);
      }
    }
  }

  return overrides;
}

function parseValue(value: string): UserConfigValue {
  if (value === 'true') {
    return true;
  }
  if (value === 'false') {
    return false;
  }
  const numValue = parseFloat(value);
  if (!isNaN(numValue) && isFinite(numValue)) {
    return numValue;
  }
  return value;
}

function loadToml(tomlContent: string): UserConfig {
  try {
    const data = toml.parse(tomlContent) as UserConfig;
    return data;
  } catch (error) {
    throw new Error('Invalid TOML content: ' + error);
  }
}
