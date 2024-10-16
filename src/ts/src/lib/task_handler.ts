import { TaskIns as ProtoTaskIns } from "../protos/flwr/proto/task";

import { PullTaskInsResponse as ProtoPullTaskInsResponse } from "../protos/flwr/proto/fleet";

export const validateTaskIns = (taskIns: ProtoTaskIns): boolean => {
  if (!(taskIns.task && taskIns.task.recordset)) {
    return false;
  }
  return true;
};

export const getTaskIns = (pullTaskInsResponse: ProtoPullTaskInsResponse): ProtoTaskIns | null => {
  if (pullTaskInsResponse.taskInsList.length === 0) {
    return null;
  }

  return pullTaskInsResponse.taskInsList[0];
};
