import { defineRailway, github, project, service } from "railway/iac";

export default defineRailway(() => {
  const handwritingLab = service("handwriting-lab", {
    source: github("haomingkoo/handwriting-lab", { checkSuites: false }),
    build: { builder: "DOCKERFILE", dockerfilePath: "inference.Containerfile" },
    replicas: { "asia-southeast1-eqsg3a": 1 },
    deploy: {
      limitOverride: { containers: { cpu: 8, memoryBytes: 8000000000 } },
      restartPolicyType: "ON_FAILURE",
      restartPolicyMaxRetries: 10,
    },
    domains: ["handwriting.kooexperience.com"],
  });

  return project("profound-essence", {
    resources: [handwritingLab],
  });
});
