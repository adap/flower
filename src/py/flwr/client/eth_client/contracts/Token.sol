// contracts/Token.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.6;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract Token is ERC20 {
    constructor() ERC20("FLWRToken", "FLWR") {
        _mint(msg.sender, 1e25);
    }
}
