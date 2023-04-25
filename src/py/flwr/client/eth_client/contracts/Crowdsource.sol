// pragma solidity >=0.4.21 <0.7.0;
pragma solidity ^0.8.6;
pragma experimental ABIEncoderV2;

/// @title Records contributions made to a Crowdsourcing Federated Learning process
/// @author Harry Cai
contract Crowdsource {
    /// @notice Address of contract creator, who evaluates updates
    address public evaluator;

    /// @notice IPFS CID of genesis model
    bytes32 public genesis;

    /// @dev The timestamp when the genesis model was upload
    uint256 internal genesisTimestamp;

    /// @notice IPFS CID of model architecture
    bytes32 public modelArchitecture;

    /// @dev Duration of each training round in seconds
    uint256 internal roundDuration;

    /// @dev Number of updates before training round automatically ends. (If 0, always wait the full roundDuration)
    uint256 internal maxNumUpdates;

    /// @dev Total number of seconds skipped when training rounds finish early
    uint256 internal timeSkipped;

    /// @dev The IPFS CIDs of model updates in each round
    mapping(uint256 => bytes32[]) internal updatesInRound;

    /// @dev The round to which each model update belongs
    mapping(bytes32 => uint256) internal updateRound;

    /// @dev The IPFS CIDs of model updates made by each address
    mapping(address => bytes32[]) internal updatesFromAddress;

    /// @dev The Account of model updates 
    mapping( bytes32 => address) internal accountsFromUpdate;

    /// @dev Whether or not each model update has been evaluated
    mapping(bytes32 => bool) internal tokensAssigned;

    /// @dev The contributivity score for each model update, if evaluated
    mapping(bytes32 => uint256) internal tokens;

    /// @dev Current Round's Trainer list
    mapping(uint256 => address[]) internal curTrainers;

    /// @dev Current Round's Evaluation complete flag
    mapping(uint256 => bool) internal evalflag;

    /// @dev Global model for each rounds
    mapping(uint256 => bytes32) internal globalmodel;

    struct ScoreandAccount {
        address account;
        int256 score;
    }

    /// @dev Account, Scores by cids 
    mapping(bytes32 => ScoreandAccount) internal score;

    event Log(string _myString , uint256 round);

    /// @notice Constructor. The address that deploys the contract is set as the evaluator.
    constructor() public {
        // evaluator = msg.sender;
        evaluator = tx.origin;
    }

    modifier evaluatorOnly() {
        // require(msg.sender == evaluator, "Not the registered evaluator");
        require(tx.origin == evaluator, "Not the registered evaluator");
        _;
    }

    function saveGlobalmodel(bytes32 _cid, uint256 _round) external {
        globalmodel[_round] = _cid;
    }

    function getGlobalmodel (uint256 _round) external view returns (bytes32){
        return globalmodel[_round];
    }

    function saveScores(bytes32 _cid, address _address, int256 _score) external evaluatorOnly(){
        score[_cid] = ScoreandAccount(_address,_score);
    }

    function getScores(bytes32 _cid) external view evaluatorOnly returns(ScoreandAccount memory){
        return score[_cid];
    }

    function completeEval(uint256 _round) external {
//        require(evalflag[_round] == false,"evaluation completed already.");
        evalflag[_round] = true;
    }

    function getCurTrainers (uint256 _round) external view returns (address[] memory  trainers){
        trainers = curTrainers[_round];
    }

    function isTrainer(address _address, uint256 _round) external view returns (bool trainCheckFlag){
        trainCheckFlag=false;
        
        for(uint i = 0; i < curTrainers[_round].length ; i++){
            if(curTrainers[_round][i] == _address){
                trainCheckFlag= true;
            }else{
                continue;
            }
        } 
        return trainCheckFlag;
    }

    function setCurTrainer(address[] memory _address, uint256 _round) public evaluatorOnly() {
        for (uint i = 0; i<_address.length; i++){
             curTrainers[_round].push(_address[i]);
        }
    }

    function changeMaxNumUpdates(uint256 _maxNum) external evaluatorOnly(){
        maxNumUpdates = _maxNum;
    }

    /// @return round The index of the current training round.
    function currentRound() public view returns (uint256 round) {
        uint256 timeElapsed = timeSkipped + block.timestamp - genesisTimestamp;
        round = 1 + (timeElapsed / roundDuration);
    }


    /// @return remaining The number of seconds remaining in the current training round.
    function secondsRemaining() public view returns (uint256 remaining) {
        uint256 timeElapsed = timeSkipped + block.timestamp - genesisTimestamp;
        remaining = roundDuration - (timeElapsed % roundDuration);
    }

    /// @return The CID's of updates in the given training round.
    function updates(uint256 _round) external view returns (bytes32[] memory) {
        return updatesInRound[_round];
    }

    /// @return count Token count of the given address up to and including the given round.
    function countTokens(address _address, uint256 _round)
        external
        view
        returns (uint256 count)
    {
        bytes32[] memory updates = updatesFromAddress[_address];
        for (uint256 i = 0; i < updates.length; i++) {
            bytes32 update = updates[i];
            if (updateRound[update] <= _round) {
                count += tokens[updates[i]];
            }
        }
    }

    /// @return count Total number of tokens up to and including the given round.
    function countTotalTokens(uint256 _round) external view returns (uint256 count) {
        for (uint256 i = 1; i <= currentRound(); i++) {
            bytes32[] memory updates = updatesInRound[i];
            for (uint256 j = 0; j < updates.length; j++) {
                bytes32 update = updates[j];
                if (updateRound[update] <= _round){
                    count += tokens[updates[j]];
                }
            }
        }
    }

    /// @return Whether the given address made a contribution in the given round.
    function madeContribution(address _address, uint256 _round)
        public
        view
        returns (bool)
    {
        for (uint256 i = 0; i < updatesFromAddress[_address].length; i++) {
            bytes32 update = updatesFromAddress[_address][i];
            if (updateRound[update] == _round) {
                return true;
            }
        }
        return false;
    }

    /// @notice Sets a new evaluator.
    function setEvaluator(address _newEvaluator) external evaluatorOnly() {
        evaluator = _newEvaluator;
    }

    /// @notice Starts training by setting the genesis model. Can only be called once.
    /// @param _cid The CID of the genesis model
    function setGenesis(
        bytes32 _cid
    ) public {
        //require(genesis == 0, "Genesis has already been set");
        genesis = _cid;
        genesisTimestamp = block.timestamp;
        roundDuration = 20000;
        maxNumUpdates = 2;
    }

    function setModelArchitecture(
        bytes32 _cid
    ) public {
        modelArchitecture = _cid;
    }

    function getModelArchitecture() external view returns (bytes32){
        require(modelArchitecture != 0, "Model architecture not exist");
        return modelArchitecture;
    }

    function getGenesis() external view returns (bytes32){
        require(genesis != 0, "Genesis model not exist");
        return genesis;
    }
    /// @notice Records a training contribution in the current round.
    function addModelUpdate(bytes32 _cid, uint256 _round) external {
        emit Log("curRound : ", currentRound());
        emit Log("inserted Round : ", _round);
        require(_round > 0, "Cannot add an update for the genesis round");
        require(
            _round >= currentRound(),
            "Cannot add an update for a past round"
        );
        require(
            _round <= currentRound(),
            "Cannot add an update for a future round"
        );
        require(
            // !madeContribution(msg.sender, _round),
            !madeContribution(tx.origin, _round),
            "Already added an update for this round"
        );

        updatesInRound[_round].push(_cid);
        // updatesFromAddress[msg.sender].push(_cid);
        // accountsFromUpdate[_cid] = msg.sender;
        updatesFromAddress[tx.origin].push(_cid);
        accountsFromUpdate[_cid] = tx.origin;
        updateRound[_cid] = _round;

        // if (
        //     maxNumUpdates > 0 && updatesInRound[_round].length >= maxNumUpdates && _round == 1
        // ) {
        //     // Skip to the end of training round
        //     timeSkipped += secondsRemaining();
        // }
    }

     function skipRound(uint256 _round) 
        external
    {
        // require( maxNumUpdates > 0 && updatesInRound[_round].length >= maxNumUpdates, "trainers did not finish their training process");
        // require( evalflag[_round] == true, "evaluation is not completed");
        // timeSkipped += secondsRemaining();
        if (
            maxNumUpdates > 0 && updatesInRound[_round].length >= maxNumUpdates && (evalflag[_round] == true || _round ==1)
        ) {
            // Skip to the end of training round
            timeSkipped += secondsRemaining();
        }
    }

    function waitTrainers (uint256 _round) external view returns (bool) {
        if(updatesInRound[_round].length >= maxNumUpdates){
            return true;
        }else{
            return false;
        }
    }

    function getmaxNum () public view returns (uint256) {
        return maxNumUpdates;
    }

    function getAccountfromUpdate (bytes32 _cid) external view returns(address ){
        return accountsFromUpdate[_cid];
    }

    /// @notice Assigns a token count to an update.
    /// @param _cid The update being rewarded
    /// @param _numTokens The number of tokens to award; should be based on marginal value contribution
    function setTokens(bytes32 _cid, uint256 _numTokens)
        external
        evaluatorOnly()
    {
        require(!tokensAssigned[_cid], "Update has already been rewarded");
        tokens[_cid] = _numTokens;
        tokensAssigned[_cid] = true;
    }
}
