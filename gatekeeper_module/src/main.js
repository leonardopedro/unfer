// S4 demo gatekeeper: resolves pending side-effecting actions on the kernel's
// approval lane. `approve` applies the first pending record for a given effect;
// `deny` rejects it instead.
export async function approve(kernel, args) {
  const actions = await kernel.uk_action_list();
  const pending = actions.find(
    (a) => a.state === "pending" && a.effect === (args.effect ?? "send_notification")
  );
  if (!pending) return { applied: false, reason: "no pending action" };
  await kernel.uk_action_apply(pending.handle);
  return { applied: true };
}

export async function deny(kernel, args) {
  const actions = await kernel.uk_action_list();
  const pending = actions.find(
    (a) => a.state === "pending" && a.effect === (args.effect ?? "send_notification")
  );
  if (!pending) return { rejected: false, reason: "no pending action" };
  await kernel.uk_action_reject(pending.handle);
  return { rejected: true };
}
