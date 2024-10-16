"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const message_handler_1 = require("./message_handler");
const typing_1 = require("./typing");
const client_1 = require("./client");
const recordset_1 = require("./recordset");
const recordset_compat_1 = require("./recordset_compat");
function removeCreatedAtField(metadata) {
    const { createdAt, ...rest } = metadata;
    return rest;
}
// Mock ClientWithoutProps and ClientWithProps
class ClientWithoutProps extends client_1.Client {
    getParameters() {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            parameters: { tensors: [], tensorType: "" },
        };
    }
    fit() {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            parameters: { tensors: [], tensorType: "" },
            numExamples: 1,
            metrics: {},
        };
    }
    evaluate() {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            loss: 1.0,
            numExamples: 1,
            metrics: {},
        };
    }
}
class ClientWithProps extends ClientWithoutProps {
    getProperties() {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            properties: { str_prop: "val", int_prop: 1 },
        };
    }
}
// Helper function to create the client_fn
const getClientFn = (client) => (context) => client;
describe("Message Handler Tests", () => {
    const createMessage = (messageType, content) => {
        return new typing_1.Message({
            runId: BigInt(123),
            messageId: "abc123",
            groupId: "some-group-id",
            srcNodeId: BigInt(0),
            dstNodeId: BigInt(1123),
            replyToMessage: "",
            ttl: 10,
            messageType,
            createdAt: 0,
        }, content, {});
    };
    const context = {
        nodeId: BigInt(1123),
        nodeConfig: {},
        state: new recordset_1.RecordSet(),
        runConfig: {},
    };
    test("Client without get_properties", () => {
        const client = new ClientWithoutProps({});
        const recordset = (0, recordset_compat_1.getPropertiesInsToRecordSet)({});
        const message = createMessage("get_properties", recordset);
        const actualMessage = (0, message_handler_1.handleLegacyMessageFromMsgType)(getClientFn(client), message, context);
        const expectedGetPropertiesRes = {
            status: {
                code: typing_1.Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message: "Client does not implement `get_properties`",
            },
            properties: {},
        };
        const expectedRs = (0, recordset_compat_1.getPropertiesResToRecordSet)(expectedGetPropertiesRes);
        const expectedMessage = new typing_1.Message({
            ...message.metadata,
            messageId: "",
            srcNodeId: BigInt(1123),
            dstNodeId: BigInt(0),
            replyToMessage: message.metadata.messageId,
            ttl: actualMessage.metadata.ttl,
        }, expectedRs, {});
        expect(actualMessage.content).toEqual(expectedMessage.content);
        expect(removeCreatedAtField(actualMessage.metadata)).toMatchObject(removeCreatedAtField(expectedMessage.metadata));
        // expect(actualMessage.metadata.createdAt).toBeGreaterThan(message.metadata.createdAt);
    });
    test("Client with get_properties", () => {
        const client = new ClientWithProps({});
        const recordset = (0, recordset_compat_1.getPropertiesInsToRecordSet)({});
        const message = createMessage("get_properties", recordset);
        const actualMessage = (0, message_handler_1.handleLegacyMessageFromMsgType)(getClientFn(client), message, context);
        const expectedGetPropertiesRes = {
            status: { code: typing_1.Code.OK, message: "Success" },
            properties: { str_prop: "val", int_prop: 1 },
        };
        const expectedRs = (0, recordset_compat_1.getPropertiesResToRecordSet)(expectedGetPropertiesRes);
        const expectedMessage = new typing_1.Message({
            ...message.metadata,
            messageId: "",
            srcNodeId: BigInt(1123),
            dstNodeId: BigInt(0),
            replyToMessage: message.metadata.messageId,
            ttl: actualMessage.metadata.ttl,
        }, expectedRs, {});
        expect(actualMessage.content).toEqual(expectedMessage.content);
        expect(removeCreatedAtField(actualMessage.metadata)).toMatchObject(removeCreatedAtField(expectedMessage.metadata));
        // expect(actualMessage.metadata.createdAt).toBeGreaterThan(message.metadata.createdAt);
    });
});
describe("Message Validation", () => {
    let inMetadata;
    let validOutMetadata;
    beforeEach(() => {
        inMetadata = {
            runId: BigInt(123),
            messageId: "qwerty",
            srcNodeId: BigInt(10),
            dstNodeId: BigInt(20),
            replyToMessage: "",
            groupId: "group1",
            ttl: 100,
            messageType: "train",
            createdAt: Date.now() - 10,
        };
        validOutMetadata = {
            runId: BigInt(123),
            messageId: "",
            srcNodeId: BigInt(20),
            dstNodeId: BigInt(10),
            replyToMessage: "qwerty",
            groupId: "group1",
            ttl: 100,
            messageType: "train",
            createdAt: Date.now(),
        };
    });
    test("Valid message", () => {
        const validMessage = new typing_1.Message(validOutMetadata, new recordset_1.RecordSet(), {});
        expect((0, message_handler_1.validateOutMessage)(validMessage, inMetadata)).toBe(true);
    });
    test("Invalid message run_id", () => {
        const msg = new typing_1.Message(validOutMetadata, new recordset_1.RecordSet(), {});
        const invalidMetadata = {
            runId: BigInt(12), // Different runId
            messageId: "qwerty",
            srcNodeId: BigInt(10),
            dstNodeId: BigInt(20),
            replyToMessage: "",
            groupId: "group1",
            ttl: 100,
            messageType: "train",
            createdAt: Date.now() - 10,
        };
        expect((0, message_handler_1.validateOutMessage)(msg, invalidMetadata)).toBe(false);
    });
});
