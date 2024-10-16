"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const task_handler_1 = require("./task_handler"); // Adjust the path as necessary
const recordset_1 = require("./recordset"); // Assuming RecordSet is in the same file
const task_1 = require("../protos/flwr/proto/task"); // Adjust the import paths for Protobuf
const fleet_1 = require("../protos/flwr/proto/fleet"); // Assuming PullTaskInsResponse is here
const serde_1 = require("./serde");
// Test for validateTaskIns: No task inside TaskIns
describe("validateTaskIns - No task", () => {
    it("should return false when task is null", () => {
        const taskIns = task_1.TaskIns.create({ task: {} });
        expect((0, task_handler_1.validateTaskIns)(taskIns)).toBe(false);
    });
});
// Test for validateTaskIns: No content inside Task
describe("validateTaskIns - No content", () => {
    it("should return false when recordset is null", () => {
        const taskIns = task_1.TaskIns.create({ task: task_1.Task.create() });
        expect((0, task_handler_1.validateTaskIns)(taskIns)).toBe(false);
    });
});
// Test for validateTaskIns: Valid TaskIns
describe("validateTaskIns - Valid TaskIns", () => {
    it("should return true when task contains a valid recordset", () => {
        const recordSet = new recordset_1.RecordSet();
        const taskIns = task_1.TaskIns.create({ task: { recordset: (0, serde_1.recordSetToProto)(recordSet) } });
        expect((0, task_handler_1.validateTaskIns)(taskIns)).toBe(true);
    });
});
// Test for getTaskIns: Empty response
describe("getTaskIns - Empty response", () => {
    it("should return null when task_ins_list is empty", () => {
        const res = fleet_1.PullTaskInsResponse.create({ taskInsList: [] });
        const taskIns = (0, task_handler_1.getTaskIns)(res);
        expect(taskIns).toBeNull();
    });
});
// Test for getTaskIns: Single TaskIns in response
describe("getTaskIns - Single TaskIns", () => {
    it("should return the task ins when task_ins_list contains one task", () => {
        const expectedTaskIns = task_1.TaskIns.create({ taskId: "123", task: task_1.Task.create() });
        const res = fleet_1.PullTaskInsResponse.create({ taskInsList: [expectedTaskIns] });
        const actualTaskIns = (0, task_handler_1.getTaskIns)(res);
        expect(actualTaskIns).toEqual(expectedTaskIns);
    });
});
// Test for getTaskIns: Multiple TaskIns in response
describe("getTaskIns - Multiple TaskIns", () => {
    it("should return the first task ins when task_ins_list contains multiple tasks", () => {
        const expectedTaskIns = task_1.TaskIns.create({ taskId: "123", task: task_1.Task.create() });
        const res = fleet_1.PullTaskInsResponse.create({
            taskInsList: [expectedTaskIns, task_1.TaskIns.create(), task_1.TaskIns.create()],
        });
        const actualTaskIns = (0, task_handler_1.getTaskIns)(res);
        expect(actualTaskIns).toEqual(expectedTaskIns);
    });
});
