// S4 demo client: proposes a side-effecting op and reads the merged result.
// `submit` returns immediately with a provisional (simulated) result; `read`
// reflects the real applied result once the gatekeeper approves the record.
export async function submit(kernel, args) {
  const handle = await kernel.uk_action_submit({
    effect: "send_notification",
    params: args.params ?? { to: "alice" },
  });
  const record = await kernel.uk_action_get(handle);
  return { handle, state: record.state, simulated: record.result.simulated };
}

export async function read(kernel, args) {
  const record = await kernel.uk_action_get(args.handle);
  return { state: record.state, applied: record.result.applied ?? null };
}
