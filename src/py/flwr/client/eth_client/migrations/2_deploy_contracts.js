var Crowdsource = artifacts.require("./Crowdsource.sol");
var Token = artifacts.require("./Token.sol");


module.exports = async function(deployer) {
  let crowdsource = deployer.deploy(Crowdsource);
  let token = deployer.deploy(Token);
  await Promise.all([crowdsource, token]);
};
