#!/usr/bin/env node

import { init } from '@flwr/create-flwr/utils/createProject.js';
import inquirer from 'inquirer';
import chalk from 'chalk';

const starterKits = ['node-ts', 'node-js', 'web-ts'];

(() => {
  console.log(chalk.yellow(`Welcome to Flower Intelligence!`));
  inquirer
    .prompt([
      {
        type: 'input',
        name: 'projectName',
        message: 'Enter your project name',
        default: 'awesome-flower-intelligence-app',
      },
      {
        type: 'list',
        name: 'prefix',
        message: 'Select a starter kit',
        choices: [...starterKits],
      },
    ])
    .then((answers) => {
      const requestedPackage = answers.prefix;

      if (!starterKits.includes(requestedPackage)) {
        throw new Error('Invalid package');
      }

      init(answers.projectName, requestedPackage);
    })
    .catch((error) => {
      if (error.isTtyError) {
        console.error('Cannot render the prompt...');
      } else {
        console.error(error.message);
      }
    });
})();
