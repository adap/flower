import { getTaskIns, validateTaskIns } from "./task_handler"; // Adjust the path as necessary
import { RecordSet } from "./recordset"; // Assuming RecordSet is in the same file
import { TaskIns, Task } from "../protos/flwr/proto/task"; // Adjust the import paths for Protobuf
import { PullTaskInsResponse } from "../protos/flwr/proto/fleet"; // Assuming PullTaskInsResponse is here
import { recordSetToProto } from "./serde";

// Test for validateTaskIns: No task inside TaskIns
describe("validateTaskIns - No task", () => {
  it("should return false when task is null", () => {
    const taskIns = TaskIns.create({ task: {} });
    expect(validateTaskIns(taskIns)).toBe(false);
  });
});

// Test for validateTaskIns: No content inside Task
describe("validateTaskIns - No content", () => {
  it("should return false when recordset is null", () => {
    const taskIns = TaskIns.create({ task: Task.create() });
    expect(validateTaskIns(taskIns)).toBe(false);
  });
});

// Test for validateTaskIns: Valid TaskIns
describe("validateTaskIns - Valid TaskIns", () => {
  it("should return true when task contains a valid recordset", () => {
    const recordSet = new RecordSet();
    const taskIns = TaskIns.create({ task: { recordset: recordSetToProto(recordSet) } });
    expect(validateTaskIns(taskIns)).toBe(true);
  });
});

// Test for getTaskIns: Empty response
describe("getTaskIns - Empty response", () => {
  it("should return null when task_ins_list is empty", () => {
    const res = PullTaskInsResponse.create({ taskInsList: [] });
    const taskIns = getTaskIns(res);
    expect(taskIns).toBeNull();
  });
});

// Test for getTaskIns: Single TaskIns in response
describe("getTaskIns - Single TaskIns", () => {
  it("should return the task ins when task_ins_list contains one task", () => {
    const expectedTaskIns = TaskIns.create({ taskId: "123", task: Task.create() });
    const res = PullTaskInsResponse.create({ taskInsList: [expectedTaskIns] });
    const actualTaskIns = getTaskIns(res);
    expect(actualTaskIns).toEqual(expectedTaskIns);
  });
});

// Test for getTaskIns: Multiple TaskIns in response
describe("getTaskIns - Multiple TaskIns", () => {
  it("should return the first task ins when task_ins_list contains multiple tasks", () => {
    const expectedTaskIns = TaskIns.create({ taskId: "123", task: Task.create() });
    const res = PullTaskInsResponse.create({
      taskInsList: [expectedTaskIns, TaskIns.create(), TaskIns.create()],
    });
    const actualTaskIns = getTaskIns(res);
    expect(actualTaskIns).toEqual(expectedTaskIns);
  });
});
