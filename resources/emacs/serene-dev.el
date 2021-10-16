;;; serene-dev --- Serene's development lib for Emacs users -*- lexical-binding: t; -*-
;;
;; Copyright (c) 2019-2021 Sameer Rahmani
;;
;; Author: Sameer Rahmani <lxsameer@gnu.org>
;; URL: https://serene-lang.org
;; Version: 0.1.0
;; Package-Requires: (projectile)
;;
;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <http://www.gnu.org/licenses/>.
;;
;;; Commentary:
;;  In order to use this library just put it somewhere that Emacs can load it
;; either via `require' or `load' and then call `serene/setup-dev-env' in your
;; init file.
;;
;; Keybindings:
;; * `s-c c' Runs `./builder compile'
;; * `s-c b' Runs `./builder build'
;;
;;; Code:


(defvar serene/compile-buffer "*compile*")


(defmacro serene/builder (command &optional buf-name)
  "Run the given COMMAND via the builder script.
Use the optional BUF-NAME as the buffer."
  (let ((buf (or buf-name (format "*%s*" command))))
    `(projectile-run-async-shell-command-in-root (format "./builder %s" ,command) ,buf)))


(defun serene/compile ()
  "Compile the project.
It will run the `./builder compile' asynchronously."
  (interactive)
  (serene/builder "compile" serene/compile-buffer))


(defun serene/build ()
  "Compile the project.
It will run the `./builder build' asynchronously."
  (interactive)
  (serene/builder "build" serene/compile-buffer))


(defun serene/build-release ()
  "Compile the project.
It will run the `./builder build-release' asynchronously."
  (interactive)
  (serene/builder "build-release" serene/compile-buffer))


(defun serene/run (args)
  "Run the project with the give ARGS.
It will run the `./builder build-release' asynchronously."
  (interactive "sRun: ")
  (serene/builder (format "run %s" args) serene/compile-buffer))


(defun serene/run-hello-world (args)
  "Run the project with the give ARGS.
It will run the `./builder build-release' asynchronously."
  (interactive "s-b . -l `pwd` docs.examples.hello_world ")
  (let ((cmd (format "run -b . -l `pwd` docs.examples.hello_world %s" args)))
    (serene/builder cmd serene/compile-buffer)))


(defun serene/run-hello-world-emit (args)
  "Run the project with the give ARGS.
It will run the `./builder build-release' asynchronously."
  (interactive (list (completing-read "Emit: " '("ast" "semantic" "slir" "mlir" "lir" "ir" "jit" "object" "target"))))
  (let ((cmd (format "run -b . -l `pwd` docs.examples.hello_world -emit %s" args)))
    (serene/builder cmd serene/compile-buffer)))


(defun serene/setup-keybindings ()
  "Setup the serene-dev keybindings."
  (interactive)
  (define-key c++-mode-map (kbd "s-c c") #'serene/compile)
  (define-key c++-mode-map (kbd "s-c b") #'serene/build)
  (define-key c++-mode-map (kbd "s-c r") #'serene/run)
  (define-key c++-mode-map (kbd "s-c e") #'serene/run-hello-world-emit))


(defun serene/format-buffer ()
  "Format the buffer if `lsp-format-buffer' is available."
  (when (and (eq major-mode 'c++-mode) (featurep 'lsp-mode))
    (lsp-format-buffer)))


(defun serene/setup-dev-env ()
  "Setup the development env of Serene."
  (interactive)
  (add-hook 'c++-mode-hook
            (lambda ()
              (require 'projectile)
              (serene/setup-keybindings)
              (add-hook 'before-save-hook #'serene/format-buffer))))


(provide 'serene-dev)
;;; serene-dev.el ends here
