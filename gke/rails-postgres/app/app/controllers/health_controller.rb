class HealthController < ApplicationController
  def index
    render json: {
      status: overall_status,
      timestamp: Time.current,
      database: database_status,
      environment: Rails.env
    }
  end

  private

  def overall_status
    db_status = database_status
    # Check if we have writer/reader setup or single database
    if db_status.key?(:writer)
      db_status[:writer][:status] == 'connected' ? 'ok' : 'error'
    else
      db_status[:primary][:status] == 'connected' ? 'ok' : 'error'
    end
  end

  def database_status
    # Check if we have multiple database configuration
    puts "Checking database status... #{has_multiple_databases?}"
    if has_multiple_databases?
      {
        writer: check_database_connection_role(:writing),
        reader: check_database_connection_role(:reading)
      }
    else
      # Single database configuration (development/test)
      {
        primary: check_database_connection_single
      }
    end
  end

  def has_multiple_databases?
    ActiveRecord::Base.configurations.configurations.select {|c| c.env_name == Rails.env }.size > 1
  rescue
    false
  end

  def check_database_connection_role(role)
    ApplicationRecord.connected_to(role: role) do
      connection = ApplicationRecord.connection
      host_info = connection.execute("SELECT inet_server_addr(), inet_server_port()").first
      server_version = connection.execute("SELECT version()").first['version']

      {
        status: 'connected',
        host: host_info['inet_server_addr'],
        port: host_info['inet_server_port'],
        version: server_version.split(' ')[0..2].join(' '),
        role: role.to_s,
        database: connection.current_database
      }
    end
  rescue StandardError => e
    {
      status: 'disconnected',
      error: e.message,
      role: role.to_s
    }
  end

  def check_database_connection_single
    connection = ApplicationRecord.connection
    # For single database, get basic connection info
    database_name = connection.execute("SELECT current_database()").first['current_database']
    server_version = connection.execute("SELECT version()").first['version']

    {
      status: 'connected',
      database: database_name,
      version: server_version.split(' ')[0..2].join(' '),
      role: 'single'
    }
  rescue StandardError => e
    {
      status: 'disconnected',
      error: e.message,
      role: 'single'
    }
  end
end
