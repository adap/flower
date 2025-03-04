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
import fs from 'fs';
import path from 'path';
import { Command } from 'commander';
import { startClientInternal } from './index';
import { getLoadClientAppFn } from './client_app';
import { parseConfigArgs, registerExitHandlers } from './utils';

export function runSupernode(): void {
  console.log('Starting Flower SuperNode');

  const program = _parseArgsRunSupernode();
  const args = program.parse(process.argv).opts();

  const rootCertificates = _getCertificates(args);
  const loadFn = getLoadClientAppFn('', args.app);

  startClientInternal(
    args.superlink,
    parseConfigArgs(args.nodeConfig ? [args.nodeConfig] : undefined),
    undefined,
    loadFn,
    undefined,
    args.insecure,
    args.maxRetries,
    args.maxWaitTime,
    undefined,
    rootCertificates ? String(rootCertificates) : undefined,
    args.flwrDir
  );

  // Graceful shutdown
  registerExitHandlers();
}

export function runClientApp(): void {
  console.log('The command `flower-client-app` has been replaced by `flower-supernode`.');
  console.log('Execute `flower-supernode --help` to learn how to use it.');
  registerExitHandlers();
}

function _getCertificates(args: any): Buffer | null {
  if (args.insecure) {
    if (args.rootCertificates) {
      throw new Error(
        "Conflicting options: The '--insecure' flag disables HTTPS, but '--root-certificates' was also specified. Please remove the '--root-certificates' option when running in insecure mode, or omit '--insecure' to use HTTPS."
      );
    }
    console.warn(
      `Option \`--insecure\` was set. Starting insecure HTTP client connected to ${args.superlink}`
    );
    return null;
  } else {
    const certPath = args.rootCertificates;
    if (!certPath) {
      return null;
    }

    const rootCertificates = fs.readFileSync(path.resolve(certPath));
    console.debug(
      `Starting secure HTTPS client connected to ${args.superlink} with the following certificates: ${certPath}.`
    );
    return rootCertificates;
  }
}

function _parseArgsRunSupernode(): Command {
  const program = new Command();
  program
    .description('Start a Flower SuperNode')
    .argument(
      '[app]',
      'Specify the path of the Flower App to load and run the `ClientApp`. ' +
        'The `pyproject.toml` file must be located in the root of this path. ' +
        'When this argument is provided, the SuperNode will exclusively respond to ' +
        'messages from the corresponding `ServerApp` by matching the FAB ID and FAB version.'
    )
    .option(
      '--flwr-dir <path>',
      'The path containing installed Flower Apps.\n' +
        'By default, this value is equal to:\n\n' +
        '- `$FLWR_HOME/` if `$FLWR_HOME` is defined\n' +
        '- `$XDG_DATA_HOME/.flwr/` if `$XDG_DATA_HOME` is defined\n' +
        '- `$HOME/.flwr/` in all other cases'
    );

  _parseArgsCommon(program);

  return program;
}

function _parseArgsCommon(program: Command): void {
  program
    .option(
      '--insecure',
      'Run the client without HTTPS. By default, the client runs with HTTPS enabled.'
    )
    .option(
      '--root-certificates <path>',
      'Specifies the path to the PEM-encoded root certificate file for establishing secure HTTPS connections.'
    )
    .option(
      '--superlink <address>',
      'SuperLink Fleet API (gRPC-rere) address (IPv4, IPv6, or a domain name)',
      '127.0.0.1:9092'
    )
    .option(
      '--max-retries <number>',
      'The maximum number of times the client will try to reconnect to the SuperLink before giving up in case of a connection error.',
      parseInt
    )
    .option(
      '--max-wait-time <number>',
      'The maximum duration before the client stops trying to connect to the SuperLink in case of connection error.',
      parseFloat
    )
    .option(
      '--node-config <config>',
      'A space-separated list of key/value pairs (separated by `=`) to configure the SuperNode. ' +
        'E.g. --node-config "key1=value1 partition-id=0 num-partitions=100"'
    );
}

// runSupernode(); // Uncomment this line to run the SuperNode
// runClientApp(); // Uncomment this line to run the ClientApp
