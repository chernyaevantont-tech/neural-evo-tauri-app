/**
 * Logger - Structured logging utility
 * 
 * Replaces console.log with a type-safe, configurable logging system.
 * Supports log levels, context prefixes, and optional data payloads.
 * 
 * @example
 * Logger.debug('GenomeBreed', 'Selected subgenome', { length: 5 });
 * Logger.info('Evolution', 'Starting generation', { generation: 1 });
 * Logger.warn('Validation', 'Shape mismatch detected', { expected: [28, 28, 3] });
 * Logger.error('Training', 'Failed to load dataset', error);
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogConfig {
    minLevel: LogLevel;
    showTimestamp: boolean;
    showContext: boolean;
}

const LOG_LEVELS: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
};

const DEFAULT_CONFIG: LogConfig = {
    minLevel: 'debug',
    showTimestamp: false,
    showContext: true,
};

let config: LogConfig = DEFAULT_CONFIG;

/**
 * Configure the logger
 */
export function configureLogger(newConfig: Partial<LogConfig>): void {
    config = { ...config, ...newConfig };
}

/**
 * Get current log level from environment or default
 */
function getLogLevel(): LogLevel {
    if (typeof process !== 'undefined' && process.env?.NODE_ENV === 'production') {
        return 'info';
    }
    return (process?.env?.LOG_LEVEL as LogLevel) || 'debug';
}

/**
 * Format timestamp for log messages
 */
function formatTimestamp(): string {
    return new Date().toISOString().slice(11, 23);
}

/**
 * Format context prefix
 */
function formatContext(context: string): string {
    return `[${context}]`;
}

/**
 * Check if a log level should be displayed
 */
function shouldLog(level: LogLevel): boolean {
    const currentLevel = getLogLevel();
    return LOG_LEVELS[level] >= LOG_LEVELS[currentLevel];
}

/**
 * Internal log function
 */
function log(
    level: LogLevel,
    context: string,
    message: string,
    data?: unknown
): void {
    if (!shouldLog(level)) {
        return;
    }

    const parts: string[] = [];

    if (config.showTimestamp) {
        parts.push(`[${formatTimestamp()}]`);
    }

    if (config.showContext && context) {
        parts.push(formatContext(context));
    }

    parts.push(message);

    const logFn = console[level as keyof typeof console] as typeof console.log;

    if (data !== undefined) {
        logFn(parts.join(' '), data);
    } else {
        logFn(parts.join(' '));
    }
}

export const Logger = {
    /**
     * Debug level logging (verbose development information)
     */
    debug(context: string, message: string, data?: unknown): void {
        log('debug', context, message, data);
    },

    /**
     * Info level logging (general information)
     */
    info(context: string, message: string, data?: unknown): void {
        log('info', context, message, data);
    },

    /**
     * Warning level logging (potential issues)
     */
    warn(context: string, message: string, data?: unknown): void {
        log('warn', context, message, data);
    },

    /**
     * Error level logging (actual errors)
     */
    error(context: string, message: string, error?: unknown): void {
        log('error', context, message, error);
    },

    /**
     * Group multiple log messages together
     */
    group(context: string, messages: Array<{ level: LogLevel; message: string; data?: unknown }>): void {
        if (!shouldLog('debug')) return;

        console.group(`[${context}]`);
        messages.forEach(({ level, message, data }) => {
            log(level, '', message, data);
        });
        console.groupEnd();
    },
};

/**
 * Create a logger instance with a fixed context
 * 
 * @example
 * const genomeLogger = createLogger('Genome');
 * genomeLogger.debug('Breeding started');
 */
export function createLogger(context: string) {
    return {
        debug: (message: string, data?: unknown) => Logger.debug(context, message, data),
        info: (message: string, data?: unknown) => Logger.info(context, message, data),
        warn: (message: string, data?: unknown) => Logger.warn(context, message, data),
        error: (message: string, error?: unknown) => Logger.error(context, message, error),
    };
}
