# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provide method to get the ip address of a network interface."""

from subprocess import check_output


def get_ip_address() -> str:
    """Return IP address."""
    ips = check_output(["hostname", "--all-ip-addresses"])
    ips_decoded = ips.decode("utf-8").split(" ")
    return ips_decoded[0]
