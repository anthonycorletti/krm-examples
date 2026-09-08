# Projects

Projects group agent tasks and are scoped to the authenticated subject and environment. Reads require platform:read; mutations require platform:verify. Other users' IDs return 404. Deleting a project marks it and its tasks deleted and cancels queued/running work; it does not immediately purge object-storage content.

Project metadata lives in SQLModel tables. Names and descriptions are validated at the API boundary. The current list returns the newest 100 projects.
