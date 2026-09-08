# Cluster command checks

Run ../bin/check. A fake local aws executable verifies command/argument forwarding, default and explicit cluster names, Auto Mode configuration, and rejection of missing subnet inputs. No Docker or AWS requests are involved. These checks do not prove live EKS provisioning or workload readiness.
