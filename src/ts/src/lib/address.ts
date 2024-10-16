import { isIP } from "net";

const IPV6 = 6;
const IPV4 = 4;

interface ParsedAddress {
  host: string;
  port: number;
  version: boolean | null;
}

export function parseAddress(address: string): ParsedAddress | null {
  try {
    const lastColonIndex = address.lastIndexOf(":");

    if (lastColonIndex === -1) {
      throw new Error("No port was provided.");
    }

    // Split the address into host and port.
    const rawHost = address.slice(0, lastColonIndex);
    const rawPort = address.slice(lastColonIndex + 1);

    const port = parseInt(rawPort, 10);

    if (port > 65535 || port < 1) {
      throw new Error("Port number is invalid.");
    }

    let host = rawHost.replace(/[\[\]]/g, ""); // Remove brackets for IPv6
    let version: boolean | null = null;

    const ipVersion = isIP(host);
    if (ipVersion === IPV6) {
      version = true;
    } else if (ipVersion === IPV4) {
      version = false;
    }

    return {
      host,
      port,
      version,
    };
  } catch (err) {
    return null;
  }
}
